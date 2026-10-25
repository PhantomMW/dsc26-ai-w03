# AI Usage Disclosure

## Tools Used

Generative AI assistance was used during development to help review the task requirements, discuss implementation approaches, draft Python code, identify potential edge cases, and prepare project documentation.

VS Code and the project terminal were used to edit the code, run tests, and inspect the results.

## How AI Assistance Was Verified

AI-suggested implementation changes were checked against the project requirements and tested locally.

The implementation was evaluated using the provided pytest suite, additional checks for chronological splitting and customer-history features, and experiments comparing the model against a prior-only dummy baseline on three synthetic-data seeds.

## Example of an Issue Reviewed

The customer-history implementation was reviewed to ensure that parcels with identical creation timestamps do not count as prior parcels for one another.

An additional review also identified that the prior-parcel count should include earlier parcels even when their failure label is missing. The implementation was updated to count all earlier parcels while counting observed failures separately.

## Leakage Prevention

The feature list excludes the parcel identifier and fields recorded after the delivery outcome, including `courier_notes`, `delivered_at`, and `failed`.

The evaluation uses a chronological train/test split, and the model is fitted only on the training partition before evaluation on the test partition.

## Limitations

Passing local tests and obtaining positive results on synthetic data do not guarantee that all hidden grading tests will pass or that the model will generalize to real-world deliveries.
