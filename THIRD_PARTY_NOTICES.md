# Third-party material

This repository has no open-source licence of its own: the coursework in it remains the author's, with all
rights reserved. The items below come from other people and are used under the licences named. The licence
texts are in `LICENSES/`.

## Dataset

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` is IBM's Telco Customer Churn sample file, copied unchanged
(only the file name differs) from IBM's public repository:

- Source: <https://github.com/IBM/telco-customer-churn-on-icp4d>, file `data/Telco-Customer-Churn.csv`,
  commit `1fd6fd7` (30 September 2019), Git blob `3de7a612d1609f25f21a455bda77948729369002`.
- Licence of that repository: Apache License 2.0 (`LICENSES/Apache-2.0.txt`). The repository has no NOTICE file.
- SHA-256 of the file: `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` (970,457 bytes), as
  recorded in `results/data_manifest.json`.
- The same table is listed on Kaggle at <https://www.kaggle.com/datasets/blastchar/telco-customer-churn>, where
  the licence field reads "Data files © Original Authors".

Three results files are derived from that file and repeat values from it: `results/split_indices.json` lists the
customer IDs in each split, and `results/oof_predictions.csv` and `results/test_predictions.csv` hold each
customer's churn label (`y`) and `MonthlyCharges` beside the predictions.

## Code

In `src/metrics.py`, the functions `_compute_midrank` and `_fast_delong` are adapted from
`compare_auc_delong_xu.py` in <https://github.com/yandexdataschool/roc_comparison>:

- MIT License, Copyright (c) 2021 Nikita Kazeev (`LICENSES/MIT-roc_comparison.txt`).
- That file states that it was adapted from <https://github.com/Netflix/vmaf>: BSD-2-Clause-Patent,
  Copyright (c) 2020 Netflix, Inc. (`LICENSES/BSD-2-Clause-Patent-vmaf.txt`).
- The method is the fast algorithm of Sun, X. and Xu, W. (2014) 'Fast implementation of DeLong's algorithm for
  comparing the areas under correlated receiver operating characteristic curves', *IEEE Signal Processing
  Letters*, 21(11), pp. 1389–1393.

## Quoted text in `references/`

The source notes in `references/` quote short passages from the works they describe, each with its reference.

- Passages from legislation.gov.uk, the Information Commissioner's Office and GOV.UK: contains public sector
  information licensed under the Open Government Licence v3.0
  (<https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>).
- Passages from Ofcom documents are Ofcom copyright; each record gives the document's title.
- The BCS Code of Conduct is © BCS, The Chartered Institute for IT, and is quoted briefly with a full reference.
