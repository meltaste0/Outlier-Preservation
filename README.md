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

### CTAB-GAN-Plus tail penalty

The Heart Disease CTAB-GAN+ run in this repository uses a tail-penalty regularizer inside the generator loss. The goal is not to change the model architecture, but to add an extra training signal that checks whether each generated continuous feature stays inside a plausible range learned from the real training data.

The implementation works in four steps:

1. After the real dataset is transformed, the synthesizer computes two reference quantiles for every continuous column. By default, these are the 5th and 95th percentiles. Those values define a soft band for the feature.
2. For every continuous feature, the code reconstructs the generated value back into the original feature scale. The reconstruction depends on how the feature was encoded:
	- General continuous columns are mapped from the generator output `u` back to the original scale with a linear rescaling from `[-1, 1]` into `[min, max]`.
	- Gaussian-mixture-style continuous columns use the generator output `u` together with the mixture component probabilities, component means, and standard deviations to reconstruct a value in the original scale.
3. The reconstructed value is compared against the quantile band. The penalty for one feature is `max(0, lower - x_hat) + max(0, x_hat - upper)`, so values inside the band contribute zero and values outside the band are penalized linearly by their distance from the nearest boundary.
4. The final tail penalty is the mean of these per-feature penalties across all continuous features. During generator training, it is multiplied by `tail_penalty_lambda` and added to the adversarial generator objective.

In other words, the generator is rewarded for producing continuous values that remain consistent with the central support of the real data, while still being trained normally by the adversarial and conditional objectives. The penalty is a soft constraint: it does not hard-clamp outputs, and it can be turned off by setting `tail_penalty_lambda` to `0`.

In the current Heart Disease script, the default `tail_penalty_lambda` is `0.1` and the quantile band is `(0.05, 0.95)`. That makes the penalty strong enough to influence training, but not so strong that it replaces the GAN objective.

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
 