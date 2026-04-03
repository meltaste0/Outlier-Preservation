# Code Explanation: Synthetic Data Quality Evaluation Framework

## Overview

Your code implements a comprehensive statistical evaluation framework that compares real (original) datasets with synthetically generated datasets. The primary goal is to assess whether synthetic data maintains the statistical properties and outlier characteristics of the original data. This is crucial for evaluating the quality of synthetic data generators, particularly those used for creating realistic tabular datasets.

---

## Core Purpose

The `comparison.py` file contains a sophisticated evaluation pipeline that measures how well synthetic data mirrors the statistical properties of real data across multiple dimensions:

1. **Class Distribution** - Do synthetic and real data have similar class proportions?
2. **Tail Behavior** - Are rare values (extreme quantiles) captured correctly?
3. **Univariate Outliers** - Are outlier rates similar across individual features?
4. **Multivariate Relationships** - Are feature correlations and interdependencies preserved?
5. **Data Validity** - Does synthetic data respect data constraints and ranges?

This is particularly important for evaluating generators like CTGAN (Conditional Tabular GAN) that create synthetic datasets for domains like healthcare (arrhythmia, thyroid disease) and cancer diagnosis.

---

## Key Components & Their Role in Outlier Statistics

### 1. **Data Loading & Preprocessing**

#### `_load_table()` & `_align_columns()`
- Loads real data from `.data` files (no headers, missing values marked as "?")
- Loads synthetic data from `.csv` files
- Ensures both datasets have identical column alignment for fair comparison
- **Outlier relevance**: Proper alignment is essential for detecting whether outliers are consistently identified across both datasets

#### `_to_numeric_frame()` & `_impute_with_real_medians()`
- Converts all columns to numeric values (coercing errors to NaN)
- Imputes missing values using medians from real data
- Applies median values consistently to synthetic data
- **Outlier relevance**: Missing values and data types affect outlier detection; imputation ensures comparability

---

### 2. **Class Distribution Analysis**

#### `_class_distribution_table()`
```
Purpose: Compare class label distributions between real and synthetic data
```

**Why it matters for outlier statistics:**
- Class imbalance is a form of "class rarity" - minority classes are statistical outliers
- Synthetic generators must preserve class proportions to maintain the natural rarity structure
- Imbalanced class distributions affect univariate outlier thresholds (different classes may have different outlier rates)

**Metrics computed:**
- Real and synthetic class counts
- Proportions for each class
- Absolute proportion differences
- **High differences** indicate the synthetic data fails to capture natural class imbalance (important for anomaly detection tasks)

---

### 3. **Tail Quantile Analysis**

#### `_tail_quantile_table()`
```
Purpose: Compare extreme values (1st, 5th, 95th, 99th percentiles)
```

**Why it's critical for outlier statistics:**
- Outliers exist in the tails of distributions
- This function measures whether synthetic data captures the extreme value behavior
- Quantiles at 0.01 and 0.99 represent the boundary region where outliers typically exist
- **If synthetic quantiles differ significantly from real quantiles**, synthetic outliers won't be realistic

**Example:**
- If real data has a 99th percentile of 500 but synthetic data has 450, synthetic outliers will be "less extreme"
- This causes synthetic outlier detection to miss genuine anomalies

---

### 4. **Univariate Outlier Detection**

#### `_iqr_outlier_rate()` - The IQR Method
```python
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
Outliers = values outside [Lower Bound, Upper Bound]
```

**Why this is fundamental to outlier statistics:**
- The Interquartile Range (IQR) method is a classical statistical approach for univariate outlier detection
- It's distribution-free (doesn't assume normality)
- Uses only quartile positions, making it robust to extreme values

#### `_univariate_outlier_table()`
```
Purpose: Compare outlier rates for each feature individually
```

**What it measures:**
1. For each feature, calculate IQR using **real data's quartiles**
2. Apply these thresholds to both real and synthetic data
3. Determine outlier rate in each dataset
4. Compute absolute difference in rates

**Why this matters:**
- If synthetic data has much higher/lower outlier rates for a feature, it's not preserving the natural sparsity/rarity structure
- A 10% difference in outlier rate could significantly impact anomaly detection systems
- Different features naturally have different outlier rates (this should be preserved)

**Example workflow:**
```
Feature: Blood Pressure
Real Data Q1: 80, Q3: 120
IQR = 40
Bounds: [20, 180]
Real outlier rate: 2.3%
Synthetic outlier rate: 5.1%
Difference: 2.8% ← Too high! Synthetic data has too many outliers
```

---

### 5. **Correlation Analysis**

#### `_mean_abs_corr_diff()`
```
Purpose: Measure difference in feature correlations using Spearman correlation
```

**Relationship to outlier statistics:**
- Outliers can distort correlations, especially in small datasets
- Multivariate outliers often appear as unusual correlation patterns
- If synthetic data has different correlations, multivariate outlier detection will fail
- Spearman (rank-based) is more robust to outliers than Pearson correlation

---

### 6. **Multivariate Outlier Detection**

#### `_multivariate_outlier_summary()` - Isolation Forest
```
Purpose: Detect outliers considering all features simultaneously
```

**How it works:**
1. Fits an Isolation Forest on real data (300 trees, 5% contamination)
2. Evaluates both real and synthetic data against this model
3. Computes outlier rates and anomaly scores

**Why Isolation Forest is ideal for outlier statistics:**
- **Handles high-dimensionality**: Works well with many features
- **Captures multivariate anomalies**: Detects unusual combinations of features even if individual features are normal
- **Efficient**: Uses tree-based partitioning, no distance calculations
- **No distribution assumptions**: Works with any data distribution

**Metrics computed:**
- **Real/Synthetic outlier rates**: % flagged as anomalous (-1 label)
- **Outlier rate gap**: Difference between real and synthetic rates (critical metric)
- **Anomaly scores**: Decision function values measuring "outlierness"
  - Negative scores → more anomalous
  - Positive scores → more normal

**Example Use Case:**
- Real data: 5% outlier rate in original dataset
- Synthetic data: 4.8% outlier rate
- Gap: 0.2% ← Good! Synthetic preserves multivariate outlier structure

---

### 7. **Nearest Neighbor Distance Analysis**

#### `_nearest_neighbor_summary()`
```
Purpose: Measure how similar synthetic samples are to real data
```

**How it works:**
1. Standardizes both datasets using real data's mean and std. dev.
2. Finds the nearest real neighbor for each synthetic sample
3. Computes distance statistics

**Outlier Statistics Relevance:**
- **Near-duplicate detection** (`nn_distance < 1e-6`): Identifies synthetic samples that are copies of real samples
- **Low mean distances**: Indicates synthetic data is "close" to real data (good fidelity)
- **High distances**: Suggest synthetic samples are outliers relative to real data distribution
- **P95 distance**: Shows extreme synthetic samples - those furthest from real data

**Interpretation:**
- High near-duplicate rate: Overfitting (generator memorizing training data)
- High mean distance + high outlier gap: Synthetic outliers are unrealistic

---

### 8. **Data Validity Checking**

#### `_validity_summary()`
```
Purpose: Ensure synthetic data satisfies domain constraints
```

**Constraint types:**
1. **Range violations**:
   - Values below real data's minimum
   - Values above real data's maximum
   - These are artificial "false outliers"

2. **Binary constraints**:
   - Features that should only be 0 or 1 but have synthetic values like 0.5
   - Creates physically impossible observations

**Outlier Statistics Importance:**
- False outliers (violations) inflate outlier detection rates artificially
- A synthetic generator producing out-of-range values creates "junk outliers"
- These confound analysis by mixing real anomalies with generation errors

---

## The Complete Evaluation Workflow

### Step-by-Step Process

```
1. LOAD DATA
   ├─ Load real data (no headers, handle "?" as missing)
   └─ Load synthetic data (CSV format)

2. ALIGN & CLEAN
   ├─ Ensure matching columns
   ├─ Convert to numeric
   └─ Impute missing values using real medians

3. COMPUTE SINGLE-METRIC TABLES
   ├─ Class distribution comparison
   ├─ Tail quantile comparison (1%, 5%, 95%, 99%)
   └─ Univariate outlier comparison (per feature)

4. COMPUTE SUMMARY METRICS
   ├─ Class distribution metrics
   ├─ Tail behavior metrics
   ├─ Univariate outlier rate gaps
   ├─ Correlation structure differences
   ├─ Multivariate outlier analysis (Isolation Forest)
   ├─ Nearest neighbor distances
   └─ Validity constraint violations

5. SAVE RESULTS
   ├─ class_distribution_comparison.csv
   ├─ tail_quantile_comparison.csv
   ├─ univariate_outlier_comparison.csv
   └─ summary_metrics.csv (single row with all metrics)

6. PRINT SUMMARY
   └─ Display all summary metrics for quick review
```

---

## Output Files & Their Interpretation

### 1. `summary_metrics.csv` - High-Level Overview
Contains single-row summary of all key metrics:

| Metric | Interpretation |
|--------|-----------------|
| `class_abs_prop_diff_mean` | Average class proportion error (lower = better) |
| `univariate_outlier_rate_gap_mean` | Average outlier rate difference across features |
| `univariate_outlier_rate_gap_max` | Maximum outlier rate difference (single worst feature) |
| `iforest_outlier_rate_gap` | Multivariate outlier difference |
| `nn_distance_mean` | Average synthetic-to-real distance (higher = more different) |
| `nn_near_duplicate_rate_dlt_1e-6` | Percentage of synthetic samples that are near-copies |
| `range_violation_rate_below_min` | % synthetic values below real minimum |
| `binary_constraint_violation_rate` | % violating binary constraints |

### 2. `univariate_outlier_comparison.csv` - Feature-Level Details
Feature-by-feature outlier analysis:
- Identifies which features have the largest outlier rate discrepancies
- Helps identify specific features where synthetic data fails

### 3. `tail_quantile_comparison.csv` - Extreme Value Analysis
Shows 1st, 5th, 95th, 99th percentiles for each feature:
- Reveals whether synthetic data captures extreme values realistically
- Large differences at 0.99 quantile indicate unrealistic outliers

### 4. `class_distribution_comparison.csv` - Class Balance Analysis
Compares class proportions:
- Shows if minority classes are preserved (important for imbalanced data)
- Highlights class generation quality

---

## Why This Matters for Outlier Statistics

### Problem Context
When you generate synthetic data (e.g., using CTGAN), you need confidence that:
1. **Outliers are realistic** - Synthetic outliers behave like real outliers
2. **Outlier rates are preserved** - Anomaly detection thresholds remain valid
3. **Relationships intact** - Multivariate anomalies are captured correctly
4. **No junk outliers** - Generation errors don't create false anomalies

### How Your Code Solves This

Your framework provides **quantitative evidence** across three outlier perspectives:

#### **Univariate Outliers** (1-feature anomalies)
- Detects single-feature anomalies
- Checks if each feature maintains its natural outlier frequency
- IQR method is interpretable and widely used

#### **Multivariate Outliers** (multi-feature anomalies)
- Captures unusual feature combinations
- Isolation Forest finds complex anomaly patterns
- More realistic than univariate detection for real-world data

#### **Distribution Tails** (boundary outliers)
- Ensures extreme values are realistically distributed
- Validates that synthetic generators don't produce impossible outliers
- Quantile analysis shows where synthetic and real distributions diverge

#### **Validity Checks** (impossible outliers)
- Prevents false outliers from data generation errors
- Ensures anomaly detection isn't confounded with garbage values

---

## Practical Example: Arrhythmia Dataset

Your code is applied to the **Arrhythmia** dataset:
- 279 features representing cardiac measurements
- 16 classes (normal heart rhythm + 15 arrhythmia types)
- Heavily imbalanced (some classes are rare - they're "outlier classes")

**What your analysis reveals:**
1. Does CTGAN-generated synthetic arrhythmia data have the same class distribution?
2. Are extreme cardiac measurements (very high/low feature values) captured?
3. Do multivariate anomalies (unusual feature combinations) appear in synthetic data?
4. Are there spurious out-of-range values in synthetic data?

This is critical because:
- Medical applications require realistic anomaly detection
- Imbalanced classes make arrhythmia a hard outlier problem
- Feature correlations encode medical knowledge (must be preserved)

---

## Key Metrics for Thesis Applications

If you're using this for thesis work, focus on these metrics:

1. **Multivariate outlier rate gap** - Most comprehensive measure
2. **Univariate outlier rate gap (max)** - Worst-case feature quality
3. **Nearest neighbor distance mean** - Overall synthetic fidelity
4. **Range/binary violation rates** - Data quality baseline

Report and discuss:
- "Synthetic data has a 1.2% lower outlier rate than real data, suggesting CTGAN slightly underrepresents anomalies"
- "The maximum outlier rate difference is 4.7% in Feature X, indicating challenges in capturing X's rare values"
- "Isolation Forest achieves 95% agreement between real and synthetic anomalies"

---

## Summary

Your code provides a rigorous framework for validating that synthetic data generators produce realistic outlier structures. It's essential for:

✓ **Thesis validation** - Quantify synthetic data quality  
✓ **Model robustness** - Ensure anomaly detection models trained on synthetic data work on real data  
✓ **Generator evaluation** - Compare different CTGAN configurations  
✓ **Publication quality** - Provide rigorous statistical evidence for claims about synthetic data quality  

The multi-faceted approach (univariate, multivariate, tail behavior, validity) ensures no aspect of outlier structure is overlooked.
