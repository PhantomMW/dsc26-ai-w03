# Tawseel — Baseline Results

## Evaluation Setup

The model was evaluated against a `DummyClassifier` using the prior strategy.

* Dataset: Synthetic Tawseel parcel data generated with `make_parcels`.
* Dataset size: 4,000 parcels per seed.
* Split: Chronological split using `created_at`, with a 20% test fraction.
* Seeds: 21, 42, and 73.
* Primary metric: PR-AUC.
* Additional metric: Brier score.

## Results

| Seed | Model PR-AUC | Dummy PR-AUC | PR-AUC Lift | Model Brier | Dummy Brier |
| ---: | -----------: | -----------: | ----------: | ----------: | ----------: |
|   21 |       0.4041 |       0.2400 |     +0.1641 |      0.1712 |      0.1824 |
|   42 |       0.3172 |       0.2137 |     +0.1035 |      0.1643 |      0.1681 |
|   73 |       0.3809 |       0.2475 |     +0.1334 |      0.1786 |      0.1868 |

## Interpretation

The logistic regression pipeline outperformed the prior-only dummy baseline on PR-AUC for all three evaluated seeds.

The PR-AUC lift was positive in every experiment, ranging from 0.1035 to 0.1641.

The model also achieved a lower Brier score than the dummy baseline in all three experiments.

## Limitations

These results use synthetic data and do not establish real-world delivery performance.

The evaluation uses a chronological split, but further validation is required before any operational deployment.

## Three-Sentence Summary for Sherif

The baseline logistic regression model outperformed the prior-only dummy baseline on PR-AUC across all three evaluated seeds.

PR-AUC lift ranged from 0.1035 to 0.1641, and the model achieved a lower Brier score in every experiment.

These results are encouraging on synthetic data, but validation on representative real-world delivery data is still needed.
