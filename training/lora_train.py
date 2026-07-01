import argparse
import copy
import gc
import logging
import math
import os
import shutil
from copy import deepcopy

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline, QwenImageTransformer2DModel
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from training.control_dataset import loader, image_resize
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from tqdm.auto import tqdm
import transformers
import datasets
import diffusers

# Optional imports - only needed if quantize=True
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    bnb = None

try:
    from optimum.quanto import quantize, qfloat8, freeze
    HAS_QUANTO = True
except ImportError:
    HAS_QUANTO = False
    quantize, qfloat8, freeze = None, None, None

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for isometric map infilling")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    return parser.parse_args().config


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


def main():
    args = OmegaConf.load(parse_args())
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Resolve paths relative to project root
    for path_key in ['img_dir', 'control_dir', 'caption_dir', 'output_dir', 'logging_dir']:
        if hasattr(args.data_config, path_key) and args.data_config[path_key]:
            args.data_config[path_key] = os.path.join(project_root, args.data_config[path_key])
        elif hasattr(args, path_key) and args.get(path_key) and not os.path.isabs(args.get(path_key, '')):
            args[path_key] = os.path.join(project_root, args[path_key])
    
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
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
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoding_pipeline = QwenImageEditPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)

    cached_text_embeddings = None
    txt_cache_dir = None
    if args.precompute_text_embeddings or args.precompute_image_embeddings:
        if accelerator.is_main_process:
            cache_dir = os.path.join(args.output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        cache_dir = os.path.join(args.output_dir, "cache")
        
        # Clear old cache BEFORE precomputing (prevents stale data issues)
        if accelerator.is_main_process and os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Cleared old cache at {cache_dir}")

    if args.precompute_text_embeddings:
        with torch.no_grad():
            if args.save_cache_on_disk:
                txt_cache_dir = os.path.join(cache_dir, "text_embs")
                os.makedirs(txt_cache_dir, exist_ok=True)
            else:
                cached_text_embeddings = {}
            
            # Batch processing for faster encoding
            batch_size = 8
            all_images = []
            all_prompts = []
            all_keys = []
            
            # prompts_dir: look for prompts here
            prompts_dir = os.path.join(os.path.dirname(args.data_config.img_dir), 'prompts')
            os.makedirs(prompts_dir, exist_ok=True)
            
            for img_name in os.listdir(args.data_config.control_dir):
                if not img_name.endswith(('.png', '.jpg')):
                    continue
                img_path = os.path.join(args.data_config.control_dir, img_name)
                img_name_key = os.path.splitext(img_name)[0]
                
                # Cache key: strip _template suffix
                if img_name_key.endswith('_template'):
                    cache_key = img_name_key[:-9]  # tile_{x}_{y}_{hash}_{mask}_{variant}
                else:
                    cache_key = img_name_key
                
                # Build tile prefix for prompt lookup
                # Control: tile_{x}_{y}_{hash}_{mask}_{variant}_template.png
                # Prompt:  tile_{x}_{y}_{hash}.txt
                if '_template' in img_name:
                    parts = img_name_key.split('_')
                    if len(parts) >= 4:
                        tile_prefix = '_'.join(parts[:4])  # tile_{x}_{y}_{hash}
                    else:
                        tile_prefix = img_name_key
                else:
                    tile_prefix = cache_key
                
                # Try prompts_dir first, then fall back to img_dir
                txt_path = None
                if os.path.exists(prompts_dir):
                    candidate = os.path.join(prompts_dir, tile_prefix + '.txt')
                    if os.path.exists(candidate):
                        txt_path = candidate
                
                if txt_path is None:
                    candidate = os.path.join(args.data_config.img_dir, tile_prefix + '.txt')
                    if os.path.exists(candidate):
                        txt_path = candidate
                
                # Skip if already cached on disk
                if args.save_cache_on_disk:
                    cache_file = os.path.join(txt_cache_dir, cache_key + '.txt.pt')
                    if os.path.exists(cache_file):
                        continue
                
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                min_dim = min(w, h)
                img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
                img = img.resize((args.data_config.img_size, args.data_config.img_size), Image.Resampling.LANCZOS)
                
                prompt = open(txt_path, encoding='utf-8').read() if txt_path else ""
                all_images.append(img)
                all_prompts.append(prompt if prompt else " ")
                all_keys.append(cache_key)
            
            # Process in batches
            for batch_start in tqdm(range(0, len(all_images), batch_size)):
                batch_end = min(batch_start + batch_size, len(all_images))
                batch_images = all_images[batch_start:batch_end]
                batch_prompts = all_prompts[batch_start:batch_end]
                batch_keys = all_keys[batch_start:batch_end]
                
                # Encode batch
                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=batch_images,
                    prompt=batch_prompts,
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                
                for i, key in enumerate(batch_keys):
                    pem = prompt_embeds[i].to('cpu')
                    pem_mask = prompt_embeds_mask[i].to('cpu') if prompt_embeds_mask is not None else None
                    
                    # Debug first batch
                    if batch_start == 0:
                        print(f"[PRECOMPUTE DEBUG] key={key}")
                        print(f"  prompt_embeds shape: {pem.shape}")
                        print(f"  prompt_embeds_mask shape: {pem_mask.shape if pem_mask is not None else 'None'}")
                    
                    emb_data = {
                        'prompt_embeds': pem,
                        'prompt_embeds_mask': pem_mask
                    }
                    empty_data = {
                        'prompt_embeds': pem.clone(),
                        'prompt_embeds_mask': pem_mask.clone() if pem_mask is not None else None
                    }
                    
                    if args.save_cache_on_disk:
                        torch.save(emb_data, os.path.join(txt_cache_dir, key + '.txt.pt'))
                        torch.save(empty_data, os.path.join(txt_cache_dir, key + '_empty.pt'))
                    else:
                        cached_text_embeddings[key + '.txt'] = emb_data
                        cached_text_embeddings[key + '.txt' + '_empty.pt'] = empty_data

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)

    cached_image_embeddings = None
    cached_image_embeddings_control = None
    
    # Image embeddings cache directory
    img_cache_dir = os.path.join(cache_dir, "image_latents") if args.save_cache_on_disk else None
    
    if args.precompute_image_embeddings:
        with torch.no_grad():
            batch_size = 8  # Process multiple images at once for faster VAE encoding
            
            # === Target images ===
            cached_image_embeddings = {}
            target_files = [f for f in os.listdir(args.data_config.img_dir) if f.endswith(('.png', '.jpg'))]
            cached_count = 0
            
            for batch_start in tqdm(range(0, len(target_files), batch_size), desc="Encoding target latents"):
                batch_end = min(batch_start + batch_size, len(target_files))
                batch_imgs = []
                batch_names = []
                batch_to_encode = []  # Only images that need encoding
                
                for img_name in target_files[batch_start:batch_end]:
                    if args.save_cache_on_disk:
                        cache_file = os.path.join(img_cache_dir, "target", img_name + '.pt')
                        if os.path.exists(cache_file):
                            cached_image_embeddings[img_name] = torch.load(cache_file)
                            cached_count += 1
                            continue
                    
                    img = Image.open(os.path.join(args.data_config.img_dir, img_name)).convert('RGB')
                    w, h = img.size
                    min_dim = min(w, h)
                    img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
                    img = img.resize((args.data_config.img_size, args.data_config.img_size), Image.Resampling.LANCZOS)
                    batch_imgs.append(img)
                    batch_names.append(img_name)
                
                if len(batch_imgs) > 0:
                    os.makedirs(os.path.join(img_cache_dir, "target"), exist_ok=True)
                    batch_arr = np.stack([(np.array(img) / 127.5) - 1 for img in batch_imgs])
                    batch_tensor = torch.from_numpy(batch_arr).permute(0, 3, 1, 2)
                    batch_tensor = batch_tensor.unsqueeze(2)
                    pixel_values = batch_tensor.to(dtype=weight_dtype).to(accelerator.device)
                    
                    latents_dist = vae.encode(pixel_values).latent_dist
                    for i, name in enumerate(batch_names):
                        sampled = latents_dist.sample()[i].to('cpu')  # [C, T, H, W]
                        sampled = sampled.squeeze(1)  # Remove T dim -> [C, H, W]
                        cached_image_embeddings[name] = sampled
                        if args.save_cache_on_disk:
                            torch.save(sampled, os.path.join(img_cache_dir, "target", name + '.pt'))
            
            if cached_image_embeddings:
                print(f"[DEBUG] Cached target latents: {len(cached_image_embeddings)} total, {cached_count} from disk, shape: {list(cached_image_embeddings.values())[0].shape}")
            
            # === Control images ===
            cached_image_embeddings_control = {}
            control_files = [f for f in os.listdir(args.data_config.control_dir) if f.endswith(('.png', '.jpg'))]
            cached_count_ctrl = 0
            
            for batch_start in tqdm(range(0, len(control_files), batch_size), desc="Encoding control latents"):
                batch_end = min(batch_start + batch_size, len(control_files))
                batch_imgs = []
                batch_names = []
                
                for img_name in control_files[batch_start:batch_end]:
                    if args.save_cache_on_disk:
                        cache_file = os.path.join(img_cache_dir, "control", img_name + '.pt')
                        if os.path.exists(cache_file):
                            cached_image_embeddings_control[img_name] = torch.load(cache_file)
                            cached_count_ctrl += 1
                            continue
                    
                    img = Image.open(os.path.join(args.data_config.control_dir, img_name)).convert('RGB')
                    w, h = img.size
                    min_dim = min(w, h)
                    img = img.crop(((w - min_dim) // 2, (h - min_dim) // 2, (w + min_dim) // 2, (h + min_dim) // 2))
                    img = img.resize((args.data_config.img_size, args.data_config.img_size), Image.Resampling.LANCZOS)
                    batch_imgs.append(img)
                    batch_names.append(img_name)
                
                if len(batch_imgs) > 0:
                    os.makedirs(os.path.join(img_cache_dir, "control"), exist_ok=True)
                    batch_arr = np.stack([(np.array(img) / 127.5) - 1 for img in batch_imgs])
                    batch_tensor = torch.from_numpy(batch_arr).permute(0, 3, 1, 2)
                    batch_tensor = batch_tensor.unsqueeze(2)
                    pixel_values = batch_tensor.to(dtype=weight_dtype).to(accelerator.device)
                    
                    latents_dist = vae.encode(pixel_values).latent_dist
                    for i, name in enumerate(batch_names):
                        sampled = latents_dist.sample()[i].to('cpu')  # [C, T, H, W]
                        sampled = sampled.squeeze(1)  # Remove T dim -> [C, H, W]
                        cached_image_embeddings_control[name] = sampled
                        if args.save_cache_on_disk:
                            torch.save(sampled, os.path.join(img_cache_dir, "control", name + '.pt'))
            
            if cached_image_embeddings_control:
                print(f"[DEBUG] Cached control latents: {len(cached_image_embeddings_control)} total, {cached_count_ctrl} from disk, shape: {list(cached_image_embeddings_control.values())[0].shape}")

        vae.to('cpu')
        torch.cuda.empty_cache()
        text_encoding_pipeline.to("cpu")
        torch.cuda.empty_cache()
    
    if not args.precompute_text_embeddings:
        del text_encoding_pipeline
        gc.collect()

    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    
    if args.quantize:
        if not HAS_QUANTO or not HAS_BNB:
            raise ImportError("quantize=True requires bitsandbytes and optimum.quanto. Set quantize: false in config or install: pip install bitsandbytes optimum-quanto")
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

    flux_transformer.to(accelerator.device)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    
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

    optimizer_cls = torch.optim.AdamW
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
        if not HAS_BNB:
            raise ImportError("adam8bit=True requires bitsandbytes. Set adam8bit: false in config or install: pip install bitsandbytes")
        optimizer = bnb.optim.Adam8bit(lora_layers, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2))
    else:
        optimizer = optimizer_cls(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # prompts_dir for loading prompts
    prompts_dir = os.path.join(os.path.dirname(args.data_config.img_dir), 'prompts')
    
    train_dataloader = loader(
        csv_path=csv_path,
        dataset_dir=os.path.dirname(csv_path),
        cached_text_embeddings=cached_text_embeddings,
        cached_image_embeddings=cached_image_embeddings,
        cached_image_embeddings_control=cached_image_embeddings_control,
        txt_cache_dir=txt_cache_dir if args.save_cache_on_disk else None,
        train_batch_size=args.data_config.train_batch_size,
        num_workers=args.data_config.num_workers,
        img_size=args.data_config.img_size,
        caption_dropout_rate=args.data_config.caption_dropout_rate,
        random_ratio=args.data_config.random_ratio,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    global_step = 0
    lora_layers_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    # Flatten data_config for easier access
    train_batch_size = args.data_config.train_batch_size if hasattr(args, 'data_config') and args.data_config is not None else getattr(args, 'train_batch_size', 1)
    num_workers = args.data_config.num_workers if hasattr(args, 'data_config') and args.data_config is not None else getattr(args, 'num_workers', 4)

    total_batch_size = train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    vae_scale_factor = 2 ** len(vae.config.temperal_downsample)

    for epoch in range(1):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                if args.precompute_text_embeddings:
                    img, prompt_embeds, prompt_embeds_mask, control_img = batch
                    print(f"[TRAIN DEBUG] prompt_embeds shape: {prompt_embeds.shape}")
                    print(f"[TRAIN DEBUG] prompt_embeds_mask shape: {prompt_embeds_mask.shape}")
                    prompt_embeds = prompt_embeds.to(dtype=weight_dtype).to(accelerator.device)
                    prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32).to(accelerator.device)
                    control_img = control_img.to(dtype=weight_dtype).to(accelerator.device)
                else:
                    img, prompts = batch

                with torch.no_grad():
                    if not args.precompute_image_embeddings:
                        if str(vae.device) == 'cpu':
                            vae.to(accelerator.device, dtype=weight_dtype)
                        pixel_values = img.to(dtype=weight_dtype).to(accelerator.device)
                        pixel_values = pixel_values.unsqueeze(2)
                        pixel_latents = vae.encode(pixel_values).latent_dist.sample()
                        vae.to('cpu')
                        torch.cuda.empty_cache()
                    else:
                        pixel_latents = img.to(dtype=weight_dtype).to(accelerator.device)

                    # Ensure pixel_latents is 5D: [B, C, 1, H, W]
                    print(f"[DEBUG] pixel_latents BEFORE dim check: {pixel_latents.shape}")
                    if pixel_latents.dim() == 4:
                        print("[DEBUG] pixel_latents is 4D, unsqueeze(2)")
                        pixel_latents = pixel_latents.unsqueeze(2)
                    elif pixel_latents.dim() == 5:
                        print("[DEBUG] pixel_latents is 5D, no change")
                    elif pixel_latents.dim() > 5:
                        pixel_latents = pixel_latents.squeeze()
                        if pixel_latents.dim() == 4:
                            pixel_latents = pixel_latents.unsqueeze(2)
                    print(f"[DEBUG] pixel_latents AFTER unsqueeze: {pixel_latents.shape}")
                    pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                    print(f"[DEBUG] pixel_latents AFTER permute: {pixel_latents.shape}")

                    # Ensure control_img is 5D: [B, C, 1, H, W]
                    if control_img.dim() == 4:
                        control_img = control_img.unsqueeze(2)
                    elif control_img.dim() > 5:
                        control_img = control_img.squeeze()
                        if control_img.dim() == 4:
                            control_img = control_img.unsqueeze(2)
                    control_img = control_img.permute(0, 2, 1, 3, 4)

                    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 1, vae.config.z_dim, 1, 1).to(
                        pixel_latents.device, pixel_latents.dtype
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

                # DEBUG: Check shapes
                print(f"[DEBUG] pixel_latents shape: {pixel_latents.shape}")  # [B, T, C, H, W]
                print(f"[DEBUG] control_img shape: {control_img.shape}")
                print(f"[DEBUG] bsz: {bsz}")

                # Squeeze T dimension for pack_latents: [B, 1, C, H, W] -> [B, C, H, W]
                pixel_latents_4d = pixel_latents.squeeze(1)
                control_img_4d = control_img.squeeze(1)
                
                print(f"[DEBUG] pixel_latents_4d shape: {pixel_latents_4d.shape}")  # [B, C, H, W]
                print(f"[DEBUG] control_img_4d shape: {control_img_4d.shape}")
                
                # Check if H and W are divisible by 2 (required by _pack_latents)
                H, W = pixel_latents_4d.shape[2], pixel_latents_4d.shape[3]
                if H % 2 != 0 or W % 2 != 0:
                    raise ValueError(f"ERROR: H={H} or W={W} not divisible by 2!")
                
                if H != W:
                    raise ValueError(f"ERROR: Non-square latent! H={H}, W={W}")
                
                packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                    pixel_latents_4d, bsz, pixel_latents_4d.shape[1], pixel_latents_4d.shape[2], pixel_latents_4d.shape[3]
                )
                packed_control_img = QwenImageEditPipeline._pack_latents(
                    control_img_4d, bsz, control_img_4d.shape[1], control_img_4d.shape[2], control_img_4d.shape[3]
                )

                # img_shapes for transformer: use original 5D shapes
                img_shapes = [[
                    (1, pixel_latents.shape[3] // 2, pixel_latents.shape[4] // 2),
                    (1, control_img.shape[3] // 2, control_img.shape[4] // 2)
                ]] * bsz
                
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

                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                target = noise - pixel_latents

                # Pack target to same space as model_pred for loss computation
                # target: [B, T, C, H, W] -> [B, C, H, W]
                target_for_loss = target.squeeze(1)
                target_packed = QwenImageEditPipeline._pack_latents(
                    target_for_loss, bsz, target_for_loss.shape[1], target_for_loss.shape[2], target_for_loss.shape[3]
                )

                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target_packed.float()) ** 2).reshape(bsz, -1),
                    1,
                )
                loss = loss.mean()

                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
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
                            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for ckpt in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

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
