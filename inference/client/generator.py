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

    Features:
        - Connection pooling for improved throughput
        - Keep-alive connections to avoid TCP handshake overhead
        - Configurable concurrency limits
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        default_steps: int = 14,
        default_guidance: float = 3.0,
        max_connections: int = 100,
        max_keepalive: int = 20,
    ):
        """
        Args:
            base_url: Base URL of inference server.
            timeout: Request timeout in seconds.
            default_steps: Default inference steps.
            default_guidance: Default guidance scale.
            max_connections: Max total connections (affects concurrency).
            max_keepalive: Max keep-alive connections (for connection reuse).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_steps = default_steps
        self.default_guidance = default_guidance
        self._client: Optional[httpx.AsyncClient] = None
        self._limits = httpx.Limits(
            max_keepalive_connections=max_keepalive,
            max_connections=max_connections,
        )

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the shared HTTP client with connection pooling."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=self._limits,
            )
        return self._client

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

        # Call API using pooled connection
        client = await self._get_client()
        response = await client.post(f"{self.base_url}/edit", json=payload)
        response.raise_for_status()
        result = response.json()

        # Decode result
        result_b64 = result["image_b64"]
        result_data = base64.b64decode(result_b64)
        return Image.open(BytesIO(result_data))

    async def health_check(self) -> dict:
        """Check server health status."""
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def list_models(self) -> dict:
        """List available models on server."""
        client = await self._get_client()
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

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


# Synchronous wrapper for non-async contexts
class SyncGenerationClient:
    """Synchronous version of GenerationClient with connection pooling."""

    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        default_steps: int = 14,
        default_guidance: float = 3.0,
        max_connections: int = 100,
        max_keepalive: int = 20,
    ):
        """
        Args:
            base_url: Base URL of inference server.
            timeout: Request timeout in seconds.
            default_steps: Default inference steps.
            default_guidance: Default guidance scale.
            max_connections: Max total connections.
            max_keepalive: Max keep-alive connections.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_steps = default_steps
        self.default_guidance = default_guidance
        self._client: Optional[httpx.Client] = None
        self._limits = httpx.Limits(
            max_keepalive_connections=max_keepalive,
            max_connections=max_connections,
        )

    def _get_client(self) -> httpx.Client:
        """Get or create the shared HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                limits=self._limits,
            )
        return self._client

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
        """Synchronous version of edit() with pooled connections."""
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

        # Call API using pooled connection
        client = self._get_client()
        response = client.post(f"{self.base_url}/edit", json=payload)
        response.raise_for_status()
        result = response.json()

        # Decode result
        result_b64 = result["image_b64"]
        result_data = base64.b64decode(result_b64)
        return Image.open(BytesIO(result_data))

    def health_check(self) -> dict:
        """Check server health."""
        client = self._get_client()
        response = client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def close(self):
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
