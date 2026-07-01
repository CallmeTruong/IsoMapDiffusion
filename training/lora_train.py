import argparse
import copy
import gc
import logging
import math
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.loaders import AttnProcsLayers
from diffusers import QwenImageEditPipeline
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
import transformers
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb

try:
    from training.control_dataset import loader, image_resize
except ModuleNotFoundError:
    from control_dataset import loader, image_resize

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()
    return args.config


class ToyDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=10):
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, 2, (num_samples,))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


def lora_processors(model):
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        return processors

    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height, None


def preprocess_image(img, pipeline, target_area=1024 * 1024):
    """Preprocess a single image for encoding."""
    calc_w, calc_h, _ = calculate_dimensions(target_area, img.size[0] / img.size[1])
    img = pipeline.image_processor.resize(img, calc_h, calc_w)
    return img


def preprocess_image_to_tensor(img, pipeline, target_area=1024 * 1024):
    """Preprocess image to normalized tensor."""
    calc_w, calc_h, _ = calculate_dimensions(target_area, img.size[0] / img.size[1])
    img = pipeline.image_processor.resize(img, calc_h, calc_w)
    arr = np.array(img)
    tensor = torch.from_numpy((arr / 127.5) - 1)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return tensor


def main():
    config_path = parse_args()
    args = OmegaConf.load(config_path)
    args.save_cache_on_disk = False
    args.precompute_text_embeddings = False
    args.precompute_image_embeddings = True

    # Resolve paths relative to config file location
    config_dir = os.path.dirname(os.path.abspath(config_path))
    
    def resolve_path(path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        return os.path.join(config_dir, path)
    
    args.data_config.img_dir = resolve_path(args.data_config.img_dir)
    args.data_config.control_dir = resolve_path(args.data_config.control_dir)
    args.data_config.prompts_dir = resolve_path(getattr(args.data_config, 'prompts_dir', None))
    args.output_dir = resolve_path(args.output_dir)
    
    print(f"[Config] img_dir: {args.data_config.img_dir}")
    print(f"[Config] control_dir: {args.data_config.control_dir}")
    print(f"[Config] output_dir: {args.output_dir}")

    # Batch sizes for precomputing
    text_batch_size = getattr(args, 'precompute_batch_size', 4)
    latent_batch_size = getattr(args, 'latent_batch_size', 4)
    print(f"[Config] Precompute batch sizes: text={text_batch_size}, latent={latent_batch_size}")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    # Load pipeline for text encoding
    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)
    
    cached_text_embeddings = None
    cached_image_embeddings = None
    cached_image_embeddings_control = None
    
    if args.precompute_text_embeddings or args.precompute_image_embeddings:
        if accelerator.is_main_process:
            cache_dir = os.path.join(args.output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        cache_dir = os.path.join(args.output_dir, "cache")
    
    # ============== TEXT EMBEDDINGS (BATCHED) ==============
    if args.precompute_text_embeddings:
        cached_text_embeddings = {}
        
        control_img_names = sorted([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or '.jpg' in i])
        total = len(control_img_names)
        
        for batch_start in tqdm(range(0, total, text_batch_size), desc="Text Embeddings"):
            batch_end = min(batch_start + text_batch_size, total)
            batch_names = control_img_names[batch_start:batch_end]
            
            with torch.no_grad():
                for img_name in batch_names:
                    img_path = os.path.join(args.data_config.control_dir, img_name)
                    txt_name = img_name.split('.')[0] + '.txt'
                    txt_path = os.path.join(args.data_config.prompts_dir, txt_name)
                    
                    print(f"[DEBUG] Looking for prompt: {txt_path}")
                    
                    if not os.path.exists(txt_path):
                        logger.warning(f"Prompt file not found: {txt_path}")
                        continue
                    
                    img = Image.open(img_path).convert('RGB')
                    prompt_image = preprocess_image(img, text_encoding_pipeline)
                    
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        prompt = f.read().strip()
                    
                    print(f"[DEBUG] Loaded prompt ({len(prompt)} chars): {prompt[:50]}...")
                    

                    txt_name = img_name.split('.')[0] + '.txt'
                    
                    # Encode with prompt
                    print(f"[DEBUG] Encoding prompt for {txt_name}...")
                    result = text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[prompt],
                        device=text_encoding_pipeline.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    
                    print(f"[DEBUG] encode_prompt result type: {type(result)}, is None: {result is None}")
                    
                    if result is None or result[0] is None or result[1] is None:
                        logger.warning(f"encode_prompt returned None/invalid for {txt_path}, skipping")
                        continue
                    
                    prompt_embeds, prompt_embeds_mask = result
                    cached_text_embeddings[txt_name] = {
                        'prompt_embeds': prompt_embeds[0].cpu(),
                        'prompt_embeds_mask': prompt_embeds_mask[0].cpu(),
                    }
                    
                    # Empty embedding
                    result_empty = text_encoding_pipeline.encode_prompt(
                        image=prompt_image,
                        prompt=[' '],
                        device=text_encoding_pipeline.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )
                    if result_empty is not None and result_empty[0] is not None and result_empty[1] is not None:
                        embeds_empty, mask_empty = result_empty
                        cached_text_embeddings[f"{txt_name}empty_embedding"] = {
                            'prompt_embeds': embeds_empty[0].cpu(),
                            'prompt_embeds_mask': mask_empty[0].cpu(),
                        }
    
    # ============== VAE LATENTS (BATCHED) ==============
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)
    
    if args.precompute_image_embeddings:
        cached_image_embeddings = {}
        
        # Target images (pixel art)
        target_img_names = sorted([i for i in os.listdir(args.data_config.img_dir) if ".png" in i or ".jpg" in i])
        total_targets = len(target_img_names)
        
        for batch_start in tqdm(range(0, total_targets, latent_batch_size), desc="Target Latents"):
            batch_end = min(batch_start + latent_batch_size, total_targets)
            batch_names = target_img_names[batch_start:batch_end]
            
            tensors = []
            with torch.no_grad():
                for img_name in batch_names:
                    img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                    tensor = preprocess_image_to_tensor(img, text_encoding_pipeline)
                    tensors.append(tensor)
                
                pixel_values = torch.cat(tensors, dim=0).to(dtype=weight_dtype, device=accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                
                for j, img_name in enumerate(batch_names):
                    cached_image_embeddings[img_name] = latents[j].cpu()
                
                del pixel_values, latents, tensors
                torch.cuda.empty_cache()
        
        # Control images
        cached_image_embeddings_control = {}
        control_img_names = sorted([i for i in os.listdir(args.data_config.control_dir) if ".png" in i or ".jpg" in i])
        total_control = len(control_img_names)
        
        for batch_start in tqdm(range(0, total_control, latent_batch_size), desc="Control Latents"):
            batch_end = min(batch_start + latent_batch_size, total_control)
            batch_names = control_img_names[batch_start:batch_end]
            
            tensors = []
            with torch.no_grad():
                for img_name in batch_names:
                    img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
                    tensor = preprocess_image_to_tensor(img, text_encoding_pipeline)
                    tensors.append(tensor)
                
                pixel_values = torch.cat(tensors, dim=0).to(dtype=weight_dtype, device=accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample()
                
                for j, img_name in enumerate(batch_names):
                    cached_image_embeddings_control[img_name] = latents[j].cpu()
                
                del pixel_values, latents, tensors
                torch.cuda.empty_cache()
        
        vae.to('cpu')
        torch.cuda.empty_cache()
        text_encoding_pipeline.to("cpu")
        torch.cuda.empty_cache()
    
    gc.collect()
    
    # ============== TRANSFORMER & TRAINING ==============
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    
    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)
        
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    flux_transformer.add_adapter(lora_config)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    flux_transformer.requires_grad_(False)
    flux_transformer.train()
    
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(n)
    
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')
    
    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))
    flux_transformer.enable_gradient_checkpointing()
    
    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    train_dataloader = loader(
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        cached_image_embeddings_control=cached_image_embeddings_control,
        **args.data_config
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    global_step = 0
    dataset1 = ToyDataset(num_samples=100, input_dim=10)
    dataloader1 = DataLoader(dataset1, batch_size=8, shuffle=True)

    lora_layers_model, optimizer, _, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, dataloader1, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=0,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    
    # Move text encoding pipeline to GPU for training
    if not args.precompute_text_embeddings:
        text_encoding_pipeline.to(accelerator.device)
    
    for epoch in range(1):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                if args.precompute_text_embeddings:
                    img, prompt_embeds, prompt_embeds_mask, control_img = batch
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype, device=accelerator.device)
                    prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32, device=accelerator.device)
                    control_img = control_img.to(dtype=weight_dtype, device=accelerator.device)
                else:
                    img, prompts, control_img = batch
                    control_img = control_img.to(dtype=weight_dtype, device=accelerator.device)
                
                with torch.no_grad():
                    if not args.precompute_image_embeddings:
                        pixel_values = img.to(dtype=weight_dtype, device=accelerator.device)
                        pixel_values = pixel_values.unsqueeze(2)
                        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                    else:
                        pixel_latents = img.to(dtype=weight_dtype, device=accelerator.device)

                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                    control_img = control_img.permute(0, 2, 1, 3, 4)
                    
                    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, 1, vae.config.z_dim, 1, 1)
                        .to(pixel_latents.device, pixel_latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
                    )
                    pixel_latents = (pixel_latents - latents_mean) * latents_std
                    control_img = (control_img - latents_mean) * latents_std

                    bsz = pixel_latents.shape[0]
                    noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme="none",
                        batch_size=bsz,
                        logit_mean=0.0,
                        logit_std=1.0,
                        mode_scale=1.29,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                
                packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                    noisy_model_input,
                    bsz, 
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                packed_control_img = QwenImageEditPipeline._pack_latents(
                    control_img,
                    bsz, 
                    control_img.shape[2],
                    control_img.shape[3],
                    control_img.shape[4],
                )
                
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz
                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)
                
                with torch.no_grad():
                    if not args.precompute_text_embeddings:
                        prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                            prompt=prompts,
                            device=packed_noisy_model_input.device,
                            num_images_per_prompt=1,
                            max_sequence_length=1024,
                        )
                    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                                logger.info(f"Removing {len(removing_checkpoints)} checkpoints")
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    try:
                        if not os.path.exists(save_path):
                            os.mkdir(save_path)
                    except:
                        pass
                    
                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )
                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
