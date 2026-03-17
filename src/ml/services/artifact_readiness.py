from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from core.data_config import Paths
from ml.candidates import interactions_candidates, popularity_fallback, transactions_candidates

REQUIRED_MANIFEST_SOURCES = ("transactions", "tx_interaction", "popularity")
STRICT_MODE_ENV_VAR = "ML_OFFLINE_ARTIFACTS_STRICT"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def get_artifact_readiness_payload(*, strict: bool = False) -> dict[str, Any]:
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
    blocking_missing_or_invalid = list(missing_or_invalid) if strict else []
    return {
        "ready": bool(ready),
        "strict_mode": bool(strict),
        "launch_ready": len(blocking_missing_or_invalid) == 0,
        "blocking_missing_or_invalid": blocking_missing_or_invalid,
        "missing_or_invalid": missing_or_invalid,
        "artifacts": artifacts,
    }


def run_check_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate offline artifact readiness.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code when artifacts are missing/invalid.",
    )
    args = parser.parse_args(argv)

    strict_mode = _resolve_strict_mode(cli_strict=bool(args.strict))
    payload = get_artifact_readiness_payload(strict=strict_mode)
    print(json.dumps(payload, ensure_ascii=True))
    if payload["missing_or_invalid"]:
        print(
            "WARNING: Offline artifacts missing/invalid; service will continue in degraded mode.",
            file=sys.stderr,
        )
    if not payload["launch_ready"]:
        raise SystemExit(1)


def _resolve_strict_mode(*, cli_strict: bool) -> bool:
    if cli_strict:
        return True
    return str(os.getenv(STRICT_MODE_ENV_VAR, "")).strip().lower() in _TRUE_VALUES


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
