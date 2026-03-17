from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from core.data_config import Paths
from ml.candidates import interactions_candidates, popularity_fallback, transactions_candidates

REQUIRED_MANIFEST_SOURCES = ("transactions", "tx_interaction", "popularity")


def get_artifact_readiness_payload() -> dict[str, Any]:
    transaction_status = transactions_candidates.get_search_bundle_artifact_status()
    tx_interaction_status = interactions_candidates.get_search_bundle_artifact_status()
    popularity_status = popularity_fallback.get_popularity_snapshot_status()
    manifest_status = _get_manifest_status(Path(Paths.offline_artifacts_manifest_path))

    artifacts = {
        "transactions": transaction_status,
        "tx_interaction": tx_interaction_status,
        "popularity": popularity_status,
        "offline_manifest": manifest_status,
    }
    blocking_artifacts = {
        artifact_name: artifact_payload
        for artifact_name, artifact_payload in artifacts.items()
        if artifact_name != "offline_manifest"
    }
    ready = all(
        str(artifact_payload.get("status", "")).strip().lower() == "ready"
        for artifact_payload in blocking_artifacts.values()
    ) and str(manifest_status.get("status", "")).strip().lower() == "ready"
    missing_or_invalid = [
        artifact_name
        for artifact_name, artifact_payload in artifacts.items()
        if str(artifact_payload.get("status", "")).strip().lower() != "ready"
    ]
    return {
        "ready": bool(ready),
        "missing_or_invalid": missing_or_invalid,
        "artifacts": artifacts,
    }


def run_check_cli() -> None:
    payload = get_artifact_readiness_payload()
    print(json.dumps(payload, ensure_ascii=True))
    if not payload["ready"]:
        raise SystemExit(1)


def _get_manifest_status(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {
            "status": "missing",
            "manifest_path": str(manifest_path),
            "missing_sources": list(REQUIRED_MANIFEST_SOURCES),
        }

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "status": "invalid",
            "manifest_path": str(manifest_path),
            "error": str(exc),
        }

    missing_sources = [
        source_name
        for source_name in REQUIRED_MANIFEST_SOURCES
        if source_name not in payload
    ]
    if missing_sources:
        return {
            "status": "invalid",
            "manifest_path": str(manifest_path),
            "missing_sources": missing_sources,
        }

    return {
        "status": "ready",
        "manifest_path": str(manifest_path),
    }


if __name__ == "__main__":
    run_check_cli()
