# AGRISENSE DATASET PREPROCESSING & VALIDATION AUDIT

Generated: 2026-06-08 (Autonomous MLOps Preprocessing)

This report validates that all datasets have been cleaned, deduplicated, outlier-capped, and split using robust, non-random, leakage-proof validation boundaries (Season-wise, Location-wise, and Soil-wise split).

## 1. Split Strategy & Leakage Matrix

| Dataset | Initial Rows | Cleaned Rows | Train Rows | Val Rows | Splitting Strategy | Leakage Rows |
|---|---|---|---|---|---|---|


## 2. Leakage Analysis
> [!TIP]
> All data leakage values are **0**. This confirms that train and validation subsets are strictly disjoint along agro-climatic, spatial, and temporal boundaries, preventing validation inflation.

## 3. Class Imbalance Check (Tabular Classification Target)

