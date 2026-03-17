from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(raw_text: str) -> str:
    digest = hashlib.sha256()
    digest.update(raw_text.encode("utf-8"))
    return digest.hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_stream:
        while True:
            chunk = input_stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_file_descriptor(path: Path) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return descriptor

    descriptor["size_bytes"] = int(path.stat().st_size)
    descriptor["mtime"] = float(path.stat().st_mtime)
    descriptor["sha256"] = sha256_file(path)
    return descriptor


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def fingerprint_rows(rows: list[dict[str, Any]], *, stable_columns: list[str]) -> str:
    if not rows:
        return sha256_text("")

    prepared_rows = []
    for row in rows:
        prepared_rows.append(
            {
                column_name: row.get(column_name)
                for column_name in stable_columns
            }
        )

    prepared_rows.sort(
        key=lambda row: tuple(
            str(row.get(column_name) or "")
            for column_name in stable_columns
        )
    )

    digest = hashlib.sha256()
    for row in prepared_rows:
        digest.update(
            json.dumps(row, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def resolve_git_commit(project_root: Path) -> str | None:
    result = _run_git(project_root, "rev-parse", "HEAD")
    if not result:
        return None
    return result.strip()


def resolve_git_branch(project_root: Path) -> str | None:
    result = _run_git(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    if not result:
        return None
    return result.strip()


def resolve_git_is_dirty(project_root: Path) -> bool | None:
    result = _run_git(project_root, "status", "--porcelain")
    if result is None:
        return None
    return bool(result.strip())


def _run_git(project_root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(project_root),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


class MlflowRunLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        tracking_uri: str,
        experiment_name: str,
        run_name: str,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tracking_uri = str(tracking_uri)
        self.experiment_name = str(experiment_name)
        self.run_name = str(run_name)
        self.tags = dict(tags or {})

        self.run_id: str | None = None
        self.artifact_uri: str | None = None
        self.error: str | None = None

        self._mlflow = None
        self._run = None

    def __enter__(self) -> "MlflowRunLogger":
        if not self.enabled:
            return self
        try:
            import mlflow

            self._mlflow = mlflow
            self._mlflow.set_tracking_uri(self.tracking_uri)
            self._mlflow.set_experiment(self.experiment_name)
            self._run = self._mlflow.start_run(run_name=self.run_name)
            self.run_id = getattr(self._run.info, "run_id", None)
            self.artifact_uri = getattr(self._run.info, "artifact_uri", None)
            for key, value in self.tags.items():
                self.set_tag(str(key), value)
        except Exception as exc:
            self.error = str(exc)
            self._mlflow = None
            self._run = None
            self.run_id = None
            self.artifact_uri = None
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._mlflow is None or self._run is None:
            return
        try:
            self._mlflow.end_run()
        except Exception:
            pass

    @property
    def active(self) -> bool:
        return self._mlflow is not None and self._run is not None

    def set_tag(self, key: str, value: Any) -> None:
        if not self.active:
            return
        try:
            self._mlflow.set_tag(str(key), _to_mlflow_value(value))
        except Exception as exc:
            self.error = str(exc)

    def log_param(self, key: str, value: Any) -> None:
        if not self.active:
            return
        try:
            self._mlflow.log_param(str(key), _to_mlflow_value(value))
        except Exception as exc:
            self.error = str(exc)

    def log_params(self, params: dict[str, Any]) -> None:
        for key, value in params.items():
            self.log_param(str(key), value)

    def log_metric(self, key: str, value: float | int) -> None:
        if not self.active:
            return
        try:
            self._mlflow.log_metric(str(key), float(value))
        except Exception as exc:
            self.error = str(exc)

    def log_metrics(self, metrics: dict[str, float | int]) -> None:
        for key, value in metrics.items():
            self.log_metric(str(key), value)

    def log_artifact(self, path: str | Path, *, artifact_path: str | None = None) -> None:
        if not self.active:
            return
        try:
            if artifact_path:
                self._mlflow.log_artifact(str(path), artifact_path=str(artifact_path))
            else:
                self._mlflow.log_artifact(str(path))
        except Exception as exc:
            self.error = str(exc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "active": bool(self.active),
            "run_id": self.run_id,
            "artifact_uri": self.artifact_uri,
            "error": self.error,
        }


def _to_mlflow_value(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "null"
    return str(value)
