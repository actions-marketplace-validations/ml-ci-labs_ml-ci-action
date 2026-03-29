# ML-CI App Upload Payload Schema

This document defines the versioned payload that `ml-ci-action` uploads to the
future ML-CI App. The upload contract is separate from `report-json`.

## Version

- Current `schema_version`: `1`
- Compatibility rule: additive fields are allowed within a schema version; any
  breaking shape change requires a new `schema_version`.

## Top-Level Shape

```json
{
  "schema_version": 1,
  "source": {
    "provider": "github-actions",
    "action_version": "0.4.1",
    "generated_at": "2026-03-29T18:42:00+00:00"
  },
  "repository": {
    "full_name": "ml-ci-labs/ml-ci-action",
    "owner": "ml-ci-labs",
    "name": "ml-ci-action"
  },
  "git": {
    "sha": "abc123",
    "ref": "refs/pull/42/merge",
    "head_ref": "feature/new-metrics",
    "base_ref": "main",
    "event_name": "pull_request",
    "pull_request_number": 42,
    "run_id": "123456789",
    "run_attempt": "1",
    "run_number": "88",
    "workflow": "ML Validation",
    "job": "validate",
    "actor": "octocat"
  },
  "model": {
    "name": "fraud-detector-v3",
    "framework": "pytorch",
    "timestamp": "2026-03-29T12:34:56Z"
  },
  "metrics": {
    "current": {
      "accuracy": 0.944,
      "loss": 0.149
    },
    "shared": ["accuracy", "loss"],
    "current_only": [],
    "baseline_only": [],
    "comparisons": [
      {
        "name": "accuracy",
        "current": 0.944,
        "baseline": 0.939,
        "delta": 0.005,
        "delta_pct": 0.0053,
        "improved": true,
        "regression": false,
        "severity": "fail",
        "tolerance": 0.02
      }
    ],
    "regression_test": {
      "method": "threshold",
      "detected": false,
      "blocking_detected": false,
      "details": {
        "tolerance": 0.02
      }
    }
  },
  "validation": {
    "validation_passed": true,
    "regression_detected": false,
    "blocking_regression_detected": false,
    "baseline_source": {
      "mode": "remote-legacy",
      "requested_ref": "main",
      "requested_path": "metrics.json",
      "resolved_ref": "main",
      "resolved_path": "metrics.json",
      "available": true,
      "reason": null
    },
    "data_policy": {
      "missing_threshold": 0.1,
      "missing_thresholds": {
        "optional_feature": 0.9
      },
      "label_column": "target",
      "include_columns": [],
      "exclude_columns": ["row_id"]
    },
    "data_validation": {
      "schema_valid": true,
      "schema_errors": [],
      "schema_warnings": [],
      "missing_value_report": {},
      "missing_value_failures": {},
      "missing_value_thresholds": {},
      "duplicate_count": 0,
      "duplicate_pct": 0.0,
      "label_column": "target",
      "label_distribution": {
        "0": 512,
        "1": 488
      },
      "baseline_label_distribution": {
        "0": 500,
        "1": 500
      },
      "label_distribution_shift": {
        "0": 0.012,
        "1": -0.012
      },
      "label_shift_detected": false,
      "drift_scores": {},
      "drift_detected": false,
      "filtered_columns": ["feature_a", "target"],
      "warnings": [],
      "failures": [],
      "details": {}
    }
  },
  "model_card": {
    "generated": true,
    "path": "/github/workspace/MODEL_CARD.md",
    "content": "# fraud-detector-v3\n..."
  },
  "lineage": {
    "dataset": {
      "name": "transactions-q1-2026",
      "version": "2026.03",
      "num_samples": 150000
    },
    "training_data_hash": "sha256:2c26b46b68ffc68ff99b453c1d30413413422b7e5f5aa0d41cf2f4e32cc4c43f",
    "model_artifact_hash": "sha256:fcde2b2edba56bf408601fb721fe9b5c338d10ee429ea04f",
    "hyperparameters": {
      "learning_rate": 0.001,
      "epochs": 50
    },
    "environment": {
      "python": "3.11.11",
      "platform": "ubuntu-latest",
      "cuda": "12.4"
    }
  }
}
```

## Field Notes

- `source`: identifies the emitting Action version and generation time.
- `repository`: stable repository identity for future repository records.
- `git`: run and pull-request context used to normalize future run records.
- `metrics`: current metrics plus baseline comparison results when available.
- `validation`: gate results and data-validation detail.
- `model_card.content`: markdown body when generated; `null` otherwise.
- `lineage.dataset`: copied from the metrics JSON `dataset` object.
- `lineage.hyperparameters`: copied from the existing top-level metrics JSON
  `hyperparameters` object.
- `lineage.training_data_hash`, `lineage.model_artifact_hash`, and
  `lineage.environment` come from the optional metrics JSON `lineage` object.

## Future DB Mapping

| Payload section | Future backend use |
|---|---|
| `repository`, `git` | repository records and run identity |
| `metrics.current`, `metrics.comparisons`, `metrics.regression_test` | metric snapshots and run summary |
| `validation` | gate outcome, data-validation evidence, and baseline provenance |
| `model_card` | persisted model-card artifact |
| `lineage` | lineage and environment evidence |

Installation identity is derived from the GitHub App auth context rather than
from explicit payload fields.
