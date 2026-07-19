"""Generation client for calling inference server."""

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Sequence

import httpx
from PIL import Image


@dataclass
class EditResult:
    """Result returned from edit() and edit_batch().

    Holds both the generated image AND the metadata needed to reproduce
    or audit the call. seed_used is the seed actually used by the model
    (random if request did not specify), time_ms is the per-item wall
    time the server measured (or elapsed/N for batched calls).
    """

    image: Image.Image
    seed_used: int
    time_ms: int


class GenerationClient:
    """
    HTTP client for calling inference server.

    Usage:
        client = GenerationClient("http://gpu-server:8000")
        result = await client.edit(template_image, prompt)

        # Batched (preferred on A100-80GB):
        results = await client.edit_batch([img1, img2], [p1, p2])

    Features:
        - Connection pooling for improved throughput
        - Keep-alive connections to avoid TCP handshake overhead
        - Configurable concurrency limits
        - Automatic fallback from /edit/batch to /edit when the server
          does not support batching (older deployments)
    """

    def __init__(
        self,
        base_url: str,
        timeout: int = 600,
        connect_timeout: int = 60,
        default_steps: int = 14,
        default_guidance: float = 3.0,
        max_connections: int = 10,
        max_keepalive: int = 5,
    ):
        """
        Args:
            base_url: Base URL of inference server.
            timeout: Request timeout in seconds.
            connect_timeout: Connection timeout in seconds (important for SSH tunnels).
            default_steps: Default inference steps.
            default_guidance: Default guidance scale.
            max_connections: Max total connections (keep low for SSH tunnels).
            max_keepalive: Max keep-alive connections (for connection reuse).
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self.default_steps = default_steps
        self.default_guidance = default_guidance
        self._client: Optional[httpx.AsyncClient] = None
        # Cached server /edit/batch capability; None = not yet queried.
        self._max_batch_size: Optional[int] = None
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
    ) -> "EditResult":
        """
        Call the /edit endpoint to generate/edit an image.

        Returns:
            EditResult(image, seed_used, time_ms). The seed in the result
            is the actual server-side seed (which may be random if the
            request did not specify one), enabling deterministic replay.
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

        # Call API using pooled connection with auto-retry for SSH tunnel drops
        client = await self._get_client()
        last_exc = None
        for attempt in range(1, 6):
            try:
                response = await client.post(f"{self.base_url}/edit", json=payload)
                response.raise_for_status()
                result = response.json()
                break
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                if attempt < 5:
                    await asyncio.sleep(2.0 * attempt)
                else:
                    raise last_exc

        # Decode result
        result_b64 = result["image_b64"]
        result_data = base64.b64decode(result_b64)
        img = Image.open(BytesIO(result_data))
        # Touch .load() to fully decode before BytesIO goes out of scope.
        img.load()
        return EditResult(
            image=img,
            seed_used=int(result["seed_used"]),
            time_ms=int(result["time_ms"]),
        )

    async def edit_batch(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: float = 2.0,
        seeds: Optional[Sequence[Optional[int]]] = None,
        prefer_batch: bool = True,
    ) -> List["EditResult"]:
        """
        Batched variant of edit(). Sends N tiles in a single HTTP request
        so the server can pack them into one pipeline call.

        Args:
            images: Input PIL images (template per tile).
            prompts: Edit prompt per tile.
            negative_prompts: Optional per-tile negative prompt.
            steps: Inference steps (default: self.default_steps).
            guidance_scale: Default guidance; broadcast to all items.
            true_cfg_scale: Default true CFG; broadcast to all items.
            seeds: Per-tile seed. None means "let server pick randomly".
            prefer_batch: If True, falls back to sequential /edit calls
                when the server reports batch=1 (older build). If False,
                will raise if /edit/batch is not supported.

        Returns:
            List of EditResult, same order as input.
        """
        if len(images) != len(prompts):
            raise ValueError(
                f"images and prompts length mismatch: "
                f"{len(images)} vs {len(prompts)}"
            )
        if len(images) == 0:
            return []

        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance

        # Decide whether to call /edit/batch. We check max_batch_size via
        # /health; if the server reports 1, we either fall back or raise.
        max_batch = await self._get_max_batch_size()
        if max_batch < 2 or not prefer_batch:
            if not prefer_batch:
                raise RuntimeError(
                    "Server does not support batching (max_batch_size<2)"
                )
            # Fall back: send items one at a time.
            return await self._edit_batch_via_single(
                images, prompts, negative_prompts, steps,
                guidance_scale, true_cfg_scale, seeds,
            )

        # Encode all images and build the per-item payload.
        items: list[dict] = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            item: dict = {
                "image_b64": base64.b64encode(buffer.getvalue()).decode(),
                "prompt": prompt,
            }
            if negative_prompts is not None and i < len(negative_prompts):
                neg = negative_prompts[i]
                if neg is not None:
                    item["negative_prompt"] = neg
            if seeds is not None and i < len(seeds) and seeds[i] is not None:
                item["seed"] = int(seeds[i])  # type: ignore[index]
            items.append(item)

        payload = {
            "items": items,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
        }

        client = await self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/edit/batch", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Server doesn't support batching -> graceful degradation.
            if e.response.status_code in (404, 405) and prefer_batch:
                self._max_batch_size = 1
                return await self._edit_batch_via_single(
                    images, prompts, negative_prompts, steps,
                    guidance_scale, true_cfg_scale, seeds,
                )
            raise

        result = response.json()
        decoded: list[EditResult] = []
        for it in result["items"]:
            data = base64.b64decode(it["image_b64"])
            img = Image.open(BytesIO(data))
            img.load()  # fully decode before BytesIO is GC'd
            decoded.append(
                EditResult(
                    image=img,
                    seed_used=int(it["seed_used"]),
                    time_ms=int(it["time_ms"]),
                )
            )
        return decoded

    async def _edit_batch_via_single(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]],
        steps: int,
        guidance_scale: float,
        true_cfg_scale: float,
        seeds: Optional[Sequence[Optional[int]]],
    ) -> List["EditResult"]:
        """Fallback: send a batch as N sequential /edit calls."""
        results: list[EditResult] = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            neg = (
                negative_prompts[i] if negative_prompts is not None
                and i < len(negative_prompts) else None
            )
            seed = (
                seeds[i] if seeds is not None and i < len(seeds) else None
            )
            res = await self.edit(
                image=img, prompt=prompt, negative_prompt=neg,
                steps=steps, guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale, seed=seed,
            )
            results.append(res)
        return results

    async def _get_max_batch_size(self) -> int:
        """Cached lookup of server's /edit/batch capability."""
        if self._max_batch_size is not None:
            return self._max_batch_size
        try:
            health = await self.health_check()
            self._max_batch_size = int(health.get("max_batch_size", 1))
        except Exception:
            # Conservative default: assume no batching.
            self._max_batch_size = 1
        return self._max_batch_size

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
        self._max_batch_size: Optional[int] = None
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
    ) -> "EditResult":
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
        img = Image.open(BytesIO(result_data))
        img.load()
        return EditResult(
            image=img,
            seed_used=int(result["seed_used"]),
            time_ms=int(result["time_ms"]),
        )

    def edit_batch(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        true_cfg_scale: float = 2.0,
        seeds: Optional[Sequence[Optional[int]]] = None,
        prefer_batch: bool = True,
    ) -> List["EditResult"]:
        """Synchronous batched variant. Same semantics as async version."""
        if len(images) != len(prompts):
            raise ValueError(
                f"images and prompts length mismatch: "
                f"{len(images)} vs {len(prompts)}"
            )
        if len(images) == 0:
            return []

        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance

        max_batch = self._get_max_batch_size()
        if max_batch < 2 or not prefer_batch:
            if not prefer_batch:
                raise RuntimeError(
                    "Server does not support batching (max_batch_size<2)"
                )
            return self._edit_batch_via_single(
                images, prompts, negative_prompts, steps,
                guidance_scale, true_cfg_scale, seeds,
            )

        items: list[dict] = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            item: dict = {
                "image_b64": base64.b64encode(buffer.getvalue()).decode(),
                "prompt": prompt,
            }
            if negative_prompts is not None and i < len(negative_prompts):
                neg = negative_prompts[i]
                if neg is not None:
                    item["negative_prompt"] = neg
            if seeds is not None and i < len(seeds) and seeds[i] is not None:
                item["seed"] = int(seeds[i])  # type: ignore[index]
            items.append(item)

        payload = {
            "items": items,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "true_cfg_scale": true_cfg_scale,
        }

        client = self._get_client()
        try:
            response = client.post(
                f"{self.base_url}/edit/batch", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 405) and prefer_batch:
                self._max_batch_size = 1
                return self._edit_batch_via_single(
                    images, prompts, negative_prompts, steps,
                    guidance_scale, true_cfg_scale, seeds,
                )
            raise

        result = response.json()
        decoded: list[EditResult] = []
        for it in result["items"]:
            data = base64.b64decode(it["image_b64"])
            img = Image.open(BytesIO(data))
            img.load()
            decoded.append(
                EditResult(
                    image=img,
                    seed_used=int(it["seed_used"]),
                    time_ms=int(it["time_ms"]),
                )
            )
        return decoded

    def _edit_batch_via_single(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        negative_prompts: Optional[Sequence[Optional[str]]],
        steps: int,
        guidance_scale: float,
        true_cfg_scale: float,
        seeds: Optional[Sequence[Optional[int]]],
    ) -> List["EditResult"]:
        results: list[EditResult] = []
        for i, (img, prompt) in enumerate(zip(images, prompts)):
            neg = (
                negative_prompts[i] if negative_prompts is not None
                and i < len(negative_prompts) else None
            )
            seed = (
                seeds[i] if seeds is not None and i < len(seeds) else None
            )
            res = self.edit(
                image=img, prompt=prompt, negative_prompt=neg,
                steps=steps, guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale, seed=seed,
            )
            results.append(res)
        return results

    def _get_max_batch_size(self) -> int:
        if self._max_batch_size is not None:
            return self._max_batch_size
        try:
            health = self.health_check()
            self._max_batch_size = int(health.get("max_batch_size", 1))
        except Exception:
            self._max_batch_size = 1
        return self._max_batch_size

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
