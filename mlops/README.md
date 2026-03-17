# DVC + MLflow Versioning

## What is tracked
- Training config: `mlops/training_config.json`
- Data/input artifacts (fingerprints): files from `version_tracking.tracked_paths`
- Model weights and metadata: `data_out/models/stage_2_3`, `data_out/models/stage_4`
- Numeric metrics for DVC: `reports/model_training_metrics.json`
- Full version manifest: `reports/model_version_manifest.json`

## Prerequisite
Install dependencies (including `mlflow`) in the project environment:
```bash
uv sync
```

## Run training with version outputs
```bash
PYTHONPATH=src uv run train-models \
  --config mlops/training_config.json \
  --emit-dvc-metrics reports/model_training_metrics.json \
  --emit-version-manifest reports/model_version_manifest.json
```

Training uses finalized recommendation impressions from PostgreSQL (`get_train_rows`), so DB must be available.

## Run through DVC
```bash
dvc repro train_models
```

## Inspect
- DVC metrics: `dvc metrics show`
- MLflow UI (local file backend):
```bash
uv run mlflow ui --backend-store-uri ./data_out/mlruns --port 5001
```

The manifest links together: git commit/dirty state, DVC files hash, tracked input hashes, training data fingerprint, model files hash, and MLflow run IDs.
