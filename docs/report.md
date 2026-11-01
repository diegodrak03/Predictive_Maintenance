# Technical report: predicting machine failures on AI4I 2020

## Problem

A milling machine processes products one after another. The goal is to predict, for each
process, whether it will end in a machine failure, so that the operator can intervene
before it happens. Missing a failure is assumed to be worse than a false alarm. Without
real costs, the operating criterion is maximum recall with precision of at least 50 % (at
most one false alarm per real failure).

## Data

AI4I 2020 (UCI, synthetic): 10,000 processes, 339 failures (3.39 %). Inputs are product
type (L, M, H), air and process temperature, rotational speed, torque and tool wear. A
failure is the OR of five modes: HDF (115), OSF (98), PWF (95), TWF (46) and RNF (19). Nine
failures have no mode flagged and are kept, because similar cases would also occur in a
real plant.

Rows form a time sequence of 120 tool cycles (tool wear goes from 0 to about 200-250 min,
then the tool is replaced). Neighbouring rows are almost identical, so a random split would
leak information between training and test.

## Method

- **Split by tool cycle.** 24 of the 120 cycles (1,951 processes, 65 failures) form a test
  set chosen with StratifiedGroupKFold and frozen until the end. Model selection uses 5-fold
  StratifiedGroupKFold on the remaining cycles.
- **Physics features.** Temperature difference (process minus air), power (torque times
  angular speed) and overstrain (tool wear times torque). Each one maps to a failure mode.
- **Leakage avoided.** UID, product ID, tool cycle and the five mode columns are never
  inputs. Encoding happens inside the pipeline.
- **Metric.** PR-AUC, because accuracy is useless at 3.4 % positives (always predicting "no
  failure" scores 96.6 %).
- **Models.** Logistic regression, random forest and XGBoost, each with and without class
  weighting, and with basic or basic + physics columns (12 configurations).

## Results

Cross-validation PR-AUC (mean ± std over 5 folds):

| Model | Basic | Basic + physics |
| --- | --- | --- |
| Logistic regression | 0.464 ± 0.052 | 0.494 ± 0.045 |
| Random forest | 0.742 ± 0.050 | **0.898 ± 0.042** |
| XGBoost | 0.746 ± 0.053 | 0.877 ± 0.033 |

Class weighting never helped. Random forest with physics features was selected (it ties
with XGBoost, which was untuned). The threshold 0.093 was chosen on out-of-fold predictions.

Frozen test set, used once:

| | Cross-validation | Test |
| --- | --- | --- |
| PR-AUC | 0.898 ± 0.042 | 0.917 |
| Precision | 0.507 | 0.496 |
| Recall | 0.883 | 0.908 |
| Detected / missed / false alarms | 242 / 32 / 235 | 59 / 6 / 60 |

Recall and PR-AUC meet the success criterion; precision misses the 50 % target by one false
alarm. The threshold was not changed after seeing the test set.

## Error analysis

- HDF, PWF and OSF are detected at 100 % in cross-validation and test. Partial dependence
  shows steps exactly at the generator's thresholds (8.6 K, 3500 W and 9000 W).
- All 32 out-of-fold misses are 24 TWF and 8 failures with no mode flagged; the 6 test misses
  are 5 TWF and 1 with no mode. TWF happens at a random time between 200 and 240 min of tool
  wear, so the model can only flag the risk zone (TWF recall 35 % in cross-validation, 44 %
  in test).
- The model is at the ceiling these inputs allow. Detecting TWF would need new information,
  such as vibration or the number of parts machined by the tool.

## Limitations

- Synthetic data with simple rules; real failures will be noisier.
- One value per process; the model assumes torque and process temperature are known when
  predicting. Using only inputs known before machining, PR-AUC falls to 0.402 ± 0.039 in
  cross-validation (0.425 on test).
- Drift: trained on the first 96 cycles and tested on the last 24, PR-AUC drops to 0.793,
  mainly because the failure mix changes (no HDF, more TWF).
- The threshold was chosen and evaluated on the same out-of-fold predictions, and it sits
  where normal processes concentrate, so precision is unstable around it.

## Recommendation for a client

Deploy the random forest with the physics features, and choose the threshold from the real
cost of a missed failure versus a false alarm. The trade-off on cross-validation (274
failures, 7,775 normal processes):

| Threshold | Precision | Recall | Detected | Missed | False alarms |
| --- | --- | --- | --- | --- | --- |
| 0.050 | 0.380 | 0.923 | 253 | 21 | 413 |
| **0.093** | **0.507** | **0.883** | **242** | **32** | **235** |
| 0.100 | 0.525 | 0.876 | 240 | 34 | 217 |
| 0.200 | 0.722 | 0.861 | 236 | 38 | 91 |
| 0.300 | 0.844 | 0.847 | 232 | 42 | 43 |
| 0.400 | 0.931 | 0.832 | 228 | 46 | 17 |
| 0.500 | 0.957 | 0.814 | 223 | 51 | 10 |

Moving from 0.093 to 0.3 loses 10 failures but avoids 192 false alarms; if a missed failure
costs about 10 false alarms, 0.3 is the better choice. In both cases: measure torque and
process temperature (they carry most of the value), monitor performance over time and
retrain when conditions change, and add a tool-condition sensor if tool wear failures are
a priority.
