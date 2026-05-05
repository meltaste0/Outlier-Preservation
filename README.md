# BA-Thesis

## Environment Quick Start

### Tabsyn
1. `cd models\tabsyn-main`
2. `conda activate tabsyn`

### TabDiff
1. `cd models\TabDiff-main`
2. `conda activate tabdiff`

### CTGAN
1. `cd models\CTGAN`
2. `conda activate ctgan`

### CTAB-GAN-Plus
1. `conda activate ctabganplus`
2. Verify Python version is `3.8.x`:
	- `python --version`
3. Install dependencies from project folder:
	- `cd models\CTAB-GAN-Plus`
	- `python -m pip install -r requirements.txt`

## Notes

- Use `-r` when installing from `requirements.txt`.
- The CTAB-GAN-Plus environment must use Python 3.8 for pinned dependency compatibility.
- In `models/CTAB-GAN-Plus/requirements.txt`, use `scikit-learn==0.24.1` (not `sklearn==0.24.1`).

## TabDiff Updates

The TabDiff code in [models/TabDiff-main/tabdiff/trainer.py](models/TabDiff-main/tabdiff/trainer.py) and [models/TabDiff-main/tabdiff/main.py](models/TabDiff-main/tabdiff/main.py) was updated to make training and sampling more reliable.

- Added gradient clipping with `torch.nn.utils.clip_grad_norm_` during the training step.
- Added a guard that skips any batch whose loss becomes non-finite, so one bad batch does not corrupt the optimizer state.
- Added final checkpoint saving at the end of training so a usable model is always written even if the run never reaches the old `best_*` save condition.
- Updated test/sample checkpoint discovery to fall back to `last_ema_model.pt` and `last_model.pt` when no `best_ema_model*` checkpoint exists.
- Kept the existing learnable noise schedule setup for `heart_disease`, but made the run more resilient to intermittent NaNs during training.

### TabDiff Usage

Train a model:

```powershell
cd models\TabDiff-main
python main.py --dataname heart_disease --mode train --no_wandb
```

Sample from the latest trained checkpoint:

```powershell
python main.py --dataname heart_disease --mode test --no_wandb
```

## COMPARISON_PLAN_BASED.PY
Measures:
1. Class distribution – absolute difference in class proportions between real and synthetic data.

2. Tail behaviour – for each continuous feature, the 1st, 5th, 95th and 99th percentiles are compared; the average absolute quantile difference quantifies tail fidelity.

3. Univariate outliers – outliers are defined as values outside [Q1−1.5 x IQR,Q3+1.5 x IQR]; the per‑feature outlier rate difference is reported.

4. Multivariate relationships – Spearman rank correlation matrices are compared via the mean absolute difference of the upper triangle entries.

5. Data validity – counts of synthetic values below the real minimum or above the real maximum, plus violations of binary constraints.

6. Resemblance (domain classifier) – Propensity Mean Squared Error (pMSE) and AUROC of a classifier trained to distinguish real from synthetic data.

7. Usability (TSTR) – logistic regression trained on synthetic data and tested on real data, reporting accuracy, F1, and AUROC.

8. Outlier‑specific utility – an Isolation Forest trained on synthetic data is used to detect anomalies in real data; recall, AUPRC, and improvement over a reference detector are measured.
 