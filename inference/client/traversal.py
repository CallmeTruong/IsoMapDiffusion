"""Tile traversal logic for quadrant-based generation."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Set, Tuple

import heapq


class TileStatus(Enum):
    """Status of a tile in the generation queue."""

    PENDING = "pending"
    GENERATING = "generating"
    DONE = "done"
    FAILED = "failed"


@dataclass
class TileInfo:
    """Information about a tile."""

    qx: int
    qy: int
    status: TileStatus = TileStatus.PENDING
    priority: float = 0.0


class TileTraversal:
    """
    Manages tile traversal order for generation.

    Uses a priority queue to generate tiles in an order that ensures
    each tile has enough context (neighboring generated tiles) for
    high-quality generation.

    Strategy:
    1. Start with seed tile(s) - generate full image
    2. Wave expansion - prioritize tiles with more generated neighbors
    3. Avoid seams - ensure neighboring tiles are generated first
    """

    def __init__(
        self,
        tiles: List[Tuple[int, int]],
        seed: Tuple[int, int] | None = None,
    ):
        """
        Initialize traversal.

        Args:
            tiles: List of (qx, qy) tile coordinates
            seed: Starting seed tile (if None, uses tile closest to origin)
        """
        self.tiles = set(tiles)
        self.tile_infos: dict[Tuple[int, int], TileInfo] = {}
        self.completed: Set[Tuple[int, int]] = set()
        self.seed = seed or self._find_seed(tiles)

        # Initialize tile infos
        for (qx, qy) in tiles:
            priority = self._calculate_priority(qx, qy)
            self.tile_infos[(qx, qy)] = TileInfo(qx, qy, TileStatus.PENDING, priority)

    def _find_seed(self, tiles: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Find the best seed tile (closest to origin or first in list)."""
        if not tiles:
            raise ValueError("No tiles provided")

        # Prefer tile closest to (0, 0) or first tile
        return min(tiles, key=lambda t: abs(t[0]) + abs(t[1]))

    def _calculate_priority(self, qx: int, qy: int) -> float:
        """
        Calculate generation priority for a tile.

        Higher priority = generate first.
        Priority is based on:
        - Number of already-generated neighbors (more = higher priority)
        - Distance from seed (closer = higher priority)
        """
        if (qx, qy) in self.completed:
            return -float("inf")

        # Count generated neighbors (4-connected)
        neighbor_count = 0
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (qx + dx, qy + dy) in self.completed:
                neighbor_count += 1

        # Distance from seed
        if self.seed:
            dist = abs(qx - self.seed[0]) + abs(qy - self.seed[1])
        else:
            dist = abs(qx) + abs(qy)

        # Priority = neighbors * 10 - distance
        # This ensures we prioritize tiles with context, but also expand outward
        return neighbor_count * 10 - dist * 0.1

    def can_generate(self, qx: int, qy: int) -> bool:
        """
        Check if a tile can be generated.

        A tile can be generated if:
        - It's not already completed
        - At least one neighboring tile has been generated (for context)
        - OR it's the seed tile (no context needed for first generation)
        """
        if (qx, qy) in self.completed:
            return False

        if (qx, qy) == self.seed:
            return True

        # Check for any generated neighbor
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            if (qx + dx, qy + dy) in self.completed:
                return True

        return False

    def get_next_batch(self, batch_size: int = 1) -> List[Tuple[int, int]]:
        """
        Get next batch of tiles to generate.

        Returns tiles in priority order (most ready first).
        """
        # Get all tiles that can be generated
        available = [
            (self.tile_infos[(qx, qy)].priority, qx, qy)
            for (qx, qy) in self.tiles
            if self.can_generate(qx, qy) and (qx, qy) not in self.completed
        ]

        if not available:
            return []

        # Sort by priority (highest first)
        available.sort(reverse=True)

        # Return top batch_size
        return [(qx, qy) for (_, qx, qy) in available[:batch_size]]

    def mark_done(self, qx: int, qy: int) -> None:
        """Mark a tile as successfully generated."""
        self.completed.add((qx, qy))
        if (qx, qy) in self.tile_infos:
            self.tile_infos[(qx, qy)].status = TileStatus.DONE

        # Update priorities of remaining tiles
        for tile in self.tiles:
            if tile not in self.completed:
                self.tile_infos[tile].priority = self._calculate_priority(*tile)

    def mark_failed(self, qx: int, qy: int) -> None:
        """Mark a tile as failed."""
        if (qx, qy) in self.tile_infos:
            self.tile_infos[(qx, qy)].status = TileStatus.FAILED

    @property
    def is_complete(self) -> bool:
        """Check if all tiles are completed."""
        return self.completed == self.tiles

    @property
    def progress(self) -> Tuple[int, int, int]:
        """Return (completed, failed, remaining) counts."""
        completed = len(self.completed)
        failed = sum(
            1
            for t in self.tiles
            if self.tile_infos[t].status == TileStatus.FAILED
        )
        remaining = len(self.tiles) - completed - failed
        return completed, failed, remaining


def quadrant_iteration_order(
    start_qx: int, start_qy: int
) -> List[Tuple[int, int, str]]:
    """
    Get the order to generate quadrants within a tile.

    Returns list of (qx, qy, region_type) tuples.

    Strategy:
    1. Generate TL quadrant first (full render - no context needed)
    2. Then TR with TL as context
    3. Then BL with TL, TR as context
    4. Finally BR with all 3 as context
    """
    return [
        (start_qx, start_qy, "tl"),  # Top-left - full render
        (start_qx + 1, start_qy, "tr"),  # Top-right - context from TL
        (start_qx, start_qy + 1, "bl"),  # Bottom-left - context from TL, TR
        (start_qx + 1, start_qy + 1, "br"),  # Bottom-right - full context
    ]


def build_neighbor_map(
    tiles: List[Tuple[int, int]]
) -> dict[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Build a map of tile -> neighboring tiles.

    Returns dict mapping each tile to its 4-connected neighbors.
    """
    neighbor_map = {tile: [] for tile in tiles}

    for tile in tiles:
        qx, qy = tile
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (qx + dx, qy + dy)
            if neighbor in neighbor_map:
                neighbor_map[tile].append(neighbor)

    return neighbor_map
