# Loan Default Prediction

Machine-learning project (Banking & Financial Services) that predicts whether a credit-card client will default next month, identifies the main risk drivers, and converts the findings into recommendations for a lender.

## Problem
Defaults create losses and non-performing assets. Predicting them before they happen supports better limit setting, pricing, early intervention and collections. Missing a defaulter (false negative) costs far more than a false alarm, so the project optimises **recall / expected cost**, not accuracy.

## Dataset
UCI *Default of Credit Card Clients* – 30,000 clients, 23 features, target `default.payment.next.month` (1 = default, ~22%).
Download from [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset) and save as `data/UCI_Credit_Card.csv`.

## Repository structure
```
├── data/                     # put UCI_Credit_Card.csv here
├── notebooks/
│   └── Loan_Default_Prediction.ipynb   # full workflow
├── images/                   # figures saved by the notebook
├── reports/                  # model_comparison.csv, test_predictions.csv (generated); put report/presentation here
├── requirements.txt
└── README.md
```

## Workflow
Data understanding → statistics → EDA (uni/bi/multivariate) → preprocessing → feature engineering → modelling → evaluation → insights → recommendations.

- **Cleaning:** undocumented `EDUCATION` (0, 5, 6) and `MARRIAGE` (0) codes merged into "others"; duplicates removed; `PAY_0` renamed `PAY_1`.
- **Feature engineering (20 features):** average bill/payment, payment-to-bill ratios, credit utilisation, max delay, number of delayed months, delay trend, bill trend.
- **Models:** Logistic Regression, Decision Tree, Random Forest, AdaBoost, KNN (class weights for imbalance, scaling inside pipelines to avoid leakage).
- **Validation:** train-vs-test gap, 5-fold stratified CV, hyper-parameter tuning, ROC/PR curves, calibration, permutation importance, cost-based threshold chosen on a validation split (test set untouched until the end).
- **Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC.

## How to run
```bash
git clone <your-repo-url>
cd loan-default-prediction
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# place data/UCI_Credit_Card.csv, then:
jupyter notebook notebooks/Loan_Default_Prediction.ipynb
```
Or open the notebook in Google Colab and upload the CSV (change the path in the first data cell).

## Results
Test set: 5,993 clients (1,326 actual defaulters). Models below use the default 0.50 threshold.

| Model | Test Recall | Test Precision | Test F1 | Test ROC-AUC | Train ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.606 | 0.446 | 0.514 | 0.758 | 0.772 |
| Decision Tree | 0.621 | 0.433 | 0.510 | 0.758 | 0.788 |
| Random Forest | 0.496 | 0.549 | 0.521 | 0.772 | 0.975 |
| AdaBoost | 0.317 | 0.685 | 0.433 | 0.772 | 0.787 |
| KNN | 0.339 | 0.629 | 0.440 | 0.756 | 1.000 |
| **Random Forest (tuned)** | **0.601** | **0.493** | **0.542** | **0.777** | n/a |

Random Forest and KNN overfit heavily (train AUC 0.98 and 1.00 vs test 0.77 and 0.76). Logistic Regression, Decision Tree and AdaBoost generalise well.

**Final model:** tuned Random Forest (200 trees, max depth 10, min leaf 20) with a cost-optimised threshold of **0.35** (assuming a missed defaulter costs 5x a false alarm).

| Final model on unseen test set | Accuracy | Precision | Recall | F1 | False negatives | False positives |
|---|---|---|---|---|---|---|
| Threshold 0.50 | 0.775 | 0.493 | 0.601 | 0.542 | 529 | 818 |
| Threshold 0.35 (chosen) | 0.630 | 0.349 | 0.780 | 0.483 | 292 | 1,926 |

At 0.35 the model catches 1,034 of 1,326 defaulters (78%), at the price of 1,926 false alarms. Under the assumed 5:1 cost the total cost improves only slightly (3,386 vs 3,463), so the threshold should be set with the bank's real cost ratio.

**Key risk factors (feature importance):** latest repayment status (`PAY_1`), maximum delay, number of months delayed, average repayment status, then `PAY_2`/`PAY_3`.

**Segment risk (test set):** clients 2+ months late in the latest cycle default 69% of the time vs about 14-15% for those paid duly or revolving; lowest credit-limit quartile 31% vs 15% for the highest; utilisation above 90% defaults 34% vs 17% below 30%.

## Business recommendations
1. Early-warning alerts for any client delaying payment in the latest cycle.
2. Risk-based limits and pricing using calibrated default probabilities; restrain limit increases for high-utilisation clients.
3. Prioritise collections by predicted risk × exposure.
4. Auto-debit and instalment plans for clients who pay only minimums.
5. Use a cost-based decision threshold agreed with Risk/Finance; review quarterly.
6. Avoid sex/marital status/age in decisions; monitor drift and fairness.

## Limitations
Single country/period (Taiwan, 2005); no income or bureau data; assumed 5:1 FN:FP cost ratio; only already-approved customers observed (selection bias); demographic features raise fairness concerns.

## Tech stack
Python, NumPy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-learn, Jupyter.
