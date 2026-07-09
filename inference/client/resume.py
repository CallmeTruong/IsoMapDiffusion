"""Spot-instance resume manifest.

When running on a spot/preemptible GPU, the pod can disappear mid-job.
This module gives the inference pipeline a place to record which tile
batches have been submitted and which have completed, so a restarted
client can pick up where it left off instead of starting over.

Manifest format (JSON, one file per job):

    {
      "version": 1,
      "job_id": "isometric-2026-07-10",
      "created_at": "2026-07-10T12:00:00Z",
      "updated_at": "2026-07-10T12:34:56Z",
      "batches": [
        {
          "batch_id": "b-0000",
          "tile_keys": ["0,0", "0,1"],
          "submitted_at": "...",
          "completed_at": "...",
          "status": "done",
          "attempts": 1,
          "output_paths": ["output/renders/tile_+0_+0_xxx.png",
                           "output/renders/tile_+0_+1_xxx.png"]
        },
        {
          "batch_id": "b-0001",
          "tile_keys": ["1,0", "1,1"],
          "submitted_at": "...",
          "status": "pending",
          "attempts": 0
        }
      ]
    }

Usage (caller side):

    manifest = ResumeManifest.load(resume_dir, job_id)
    for batch in batches:
        if manifest.is_batch_done(batch.batch_id):
            continue
        results = await client.edit_batch(...)
        manifest.mark_done(batch.batch_id, tile_keys, output_paths)
    manifest.save()
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


MANIFEST_VERSION = 1
STATUS_PENDING = "pending"
STATUS_INFLIGHT = "inflight"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now_iso() -> str:
    """ISO 8601 UTC timestamp without microseconds."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class BatchRecord:
    """One batch in the resume manifest."""

    batch_id: str
    tile_keys: List[str]
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    status: str = STATUS_PENDING
    attempts: int = 0
    last_error: Optional[str] = None
    output_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BatchRecord":
        return cls(**data)


@dataclass
class ResumeManifest:
    """Persistent record of what's done so we can resume a spot job."""

    job_id: str
    resume_dir: Path
    batches: List[BatchRecord] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    version: int = MANIFEST_VERSION

    @property
    def path(self) -> Path:
        return self.resume_dir / f"resume_{self.job_id}.json"

    # --- lifecycle ---

    @classmethod
    def load(cls, resume_dir: Path, job_id: str) -> "ResumeManifest":
        """Load an existing manifest or create a new one on disk.

        Always returns a usable object. If the file does not exist we
        create a fresh manifest; if it does we hydrate it.
        """
        resume_dir = Path(resume_dir)
        resume_dir.mkdir(parents=True, exist_ok=True)
        path = resume_dir / f"resume_{job_id}.json"

        if not path.exists():
            return cls(job_id=job_id, resume_dir=resume_dir)

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        version = int(data.get("version", MANIFEST_VERSION))
        if version != MANIFEST_VERSION:
            # We don't try to migrate older manifests; rename and start fresh.
            archive = path.with_suffix(f".v{version}.json")
            if not archive.exists():
                path.rename(archive)
            return cls(job_id=job_id, resume_dir=resume_dir)

        batches = [BatchRecord.from_dict(b) for b in data.get("batches", [])]
        return cls(
            job_id=job_id,
            resume_dir=resume_dir,
            batches=batches,
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            version=version,
        )

    def save(self) -> None:
        """Atomically write the manifest. Safe to call after every batch."""
        self.updated_at = _now_iso()
        self.resume_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        payload = {
            "version": self.version,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "batches": [b.to_dict() for b in self.batches],
        }
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        # Atomic replace: works on Windows (os.replace is atomic on same FS).
        os.replace(tmp, self.path)

    # --- batch helpers ---

    def ensure_batch(self, batch_id: str, tile_keys: List[str]) -> BatchRecord:
        """Get or create a batch record. Idempotent on restart."""
        for b in self.batches:
            if b.batch_id == batch_id:
                return b
        record = BatchRecord(batch_id=batch_id, tile_keys=list(tile_keys))
        self.batches.append(record)
        return record

    def is_batch_done(self, batch_id: str) -> bool:
        for b in self.batches:
            if b.batch_id == batch_id:
                return b.status == STATUS_DONE
        return False

    def mark_inflight(self, batch_id: str) -> None:
        rec = next(
            (b for b in self.batches if b.batch_id == batch_id), None
        )
        if rec is None:
            return
        rec.status = STATUS_INFLIGHT
        rec.submitted_at = _now_iso()
        rec.attempts += 1
        self.save()

    def mark_done(
        self,
        batch_id: str,
        output_paths: Optional[List[str]] = None,
    ) -> None:
        rec = next(
            (b for b in self.batches if b.batch_id == batch_id), None
        )
        if rec is None:
            return
        rec.status = STATUS_DONE
        rec.completed_at = _now_iso()
        if output_paths is not None:
            rec.output_paths = list(output_paths)
        rec.last_error = None
        self.save()

    def mark_failed(self, batch_id: str, error: str) -> None:
        rec = next(
            (b for b in self.batches if b.batch_id == batch_id), None
        )
        if rec is None:
            return
        rec.status = STATUS_FAILED
        rec.last_error = error
        self.save()

    # --- stats ---

    def summary(self) -> dict:
        total = len(self.batches)
        done = sum(1 for b in self.batches if b.status == STATUS_DONE)
        failed = sum(
            1 for b in self.batches if b.status == STATUS_FAILED
        )
        return {
            "job_id": self.job_id,
            "total_batches": total,
            "done": done,
            "pending": total - done - failed,
            "failed": failed,
        }
