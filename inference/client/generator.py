"""Generation client for calling inference server."""

import base64
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image


class GenerationClient:
    """
    HTTP client for calling inference server.

    Usage:
        client = GenerationClient("http://gpu-server:8000")
        result = await client.edit(template_image, prompt)
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        default_steps: int = 14,
        default_guidance: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_steps = default_steps
        self.default_guidance = default_guidance

    async def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: float = 2.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Call the /edit endpoint to generate/edit an image.

        Args:
            image: Input PIL Image (template)
            prompt: Edit instruction
            negative_prompt: What to avoid
            steps: Inference steps (default: self.default_steps)
            guidance_scale: Guidance scale (default: self.default_guidance)
            true_cfg_scale: True CFG scale
            seed: Random seed

        Returns:
            Generated PIL Image
        """
        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance

        # Encode image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Build request
        payload = {
            "image_b64": image_b64,
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }

        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt

        if seed is not None:
            payload["seed"] = seed

        # Call API
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/edit", json=payload)
            response.raise_for_status()
            result = response.json()

        # Decode result
        result_b64 = result["image_b64"]
        result_data = base64.b64decode(result_b64)
        return Image.open(BytesIO(result_data))

    async def health_check(self) -> dict:
        """Check server health status."""
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()

    async def list_models(self) -> dict:
        """List available models on server."""
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
            return response.json()

    async def wait_for_server(self, timeout: float = 60.0) -> bool:
        """
        Wait for server to be ready.

        Returns True if server is ready, False if timeout.
        """
        import asyncio

        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout:
            try:
                health = await self.health_check()
                if health.get("model_loaded"):
                    return True
            except Exception:
                pass
            await asyncio.sleep(2)

        return False


# Synchronous wrapper for non-async contexts
class SyncGenerationClient:
    """Synchronous version of GenerationClient."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        default_steps: int = 14,
        default_guidance: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_steps = default_steps
        self.default_guidance = default_guidance
        self._client = httpx.Client(timeout=timeout)

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: float = 2.0,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Synchronous version of edit()."""
        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance

        # Encode image
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()

        # Build request
        payload = {
            "image_b64": image_b64,
            "prompt": prompt,
            "true_cfg_scale": true_cfg_scale,
            "steps": steps,
            "guidance_scale": guidance_scale,
        }

        if negative_prompt is not None:
            payload["negative_prompt"] = negative_prompt

        if seed is not None:
            payload["seed"] = seed

        # Call API
        response = self._client.post(f"{self.base_url}/edit", json=payload)
        response.raise_for_status()
        result = response.json()

        # Decode result
        result_b64 = result["image_b64"]
        result_data = base64.b64decode(result_b64)
        return Image.open(BytesIO(result_data))

    def health_check(self) -> dict:
        """Check server health."""
        response = self._client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
