"""
AGRISENSE MLOps Engine - Statistical Data Drift Detector
Implements Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
"""

import numpy as np

try:
    from scipy.stats import ks_2samp

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class DriftDetector:
    @staticmethod
    def calculate_psi(
        expected: np.ndarray, actual: np.ndarray, num_bins: int = 10
    ) -> float:
        """
        Calculates the Population Stability Index (PSI) between expected and actual distributions.
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        Interpretation:
            PSI < 0.1: No significant change / stable.
            0.1 <= PSI < 0.25: Moderate shift / warning.
            PSI >= 0.25: Significant shift / action required.
        """
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Calculate percentiles/bins on expected distribution
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(expected, percentiles)
        # Handle duplicate bin edges
        bins = np.unique(bins)
        if len(bins) < 2:
            return 0.0  # Single valued feature

        # Calculate frequency counts in each bin
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        # Convert to percentages
        expected_pcts = expected_counts / len(expected)
        actual_pcts = actual_counts / len(actual)

        # Zero handling to avoid divide by zero or log of zero
        expected_pcts = np.where(expected_pcts == 0, 0.0001, expected_pcts)
        actual_pcts = np.where(actual_pcts == 0, 0.0001, actual_pcts)

        # Calculate PSI
        psi_value = np.sum(
            (actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)
        )
        return float(psi_value)

    @staticmethod
    def calculate_ks(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
        """
        Calculates the Kolmogorov-Smirnov (KS) 2-sample statistic and p-value.
        If scipy is missing, falls back to a custom NumPy empirical CDF comparison.
        """
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0, 1.0

        if SCIPY_AVAILABLE:
            res = ks_2samp(expected, actual)
            return float(res.statistic), float(res.pvalue)

        # Pure NumPy Fallback for Kolmogorov-Smirnov statistic
        # Combine data to find sorted values
        combined = np.sort(np.concatenate([expected, actual]))

        # Empirical CDF calculations
        cdf_exp = np.searchsorted(np.sort(expected), combined, side="right") / len(
            expected
        )
        cdf_act = np.searchsorted(np.sort(actual), combined, side="right") / len(actual)

        # KS statistic is max absolute difference
        ks_stat = np.max(np.abs(cdf_exp - cdf_act))

        # Simple asymptotic p-value approximation
        n1 = len(expected)
        n2 = len(actual)
        en = np.sqrt((n1 * n2) / (n1 + n2))
        # p-val approximation using Kolmogorov distribution asymptotic formula
        lambda_val = (en + 0.12 + 0.11 / en) * ks_stat
        if lambda_val < 0.2:
            p_val = 1.0
        else:
            p_val = 2 * np.exp(-2 * lambda_val**2)
            p_val = min(1.0, max(0.0, p_val))

        return float(ks_stat), float(p_val)

    def analyze_dataset_drift(self, reference_df: dict, current_df: dict) -> dict:
        """
        Computes drift metrics for numerical telemetry fields.
        Input arguments are dictionaries mapping feature names to lists/arrays.
        """
        drift_report = {}
        global_drift_detected = False
        significant_features_count = 0

        features = set(reference_df.keys()).intersection(set(current_df.keys()))

        for feat in features:
            ref_data = np.array(reference_df[feat], dtype=float)
            curr_data = np.array(current_df[feat], dtype=float)

            psi = self.calculate_psi(ref_data, curr_data)
            ks_stat, p_val = self.calculate_ks(ref_data, curr_data)

            # Drift flag conditions: PSI > 0.25 or KS p-value < 0.05
            drift_detected = (psi >= 0.2) or (p_val < 0.05)
            if drift_detected:
                global_drift_detected = True
                significant_features_count += 1

            drift_report[feat] = {
                "psi_score": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_p_value": round(p_val, 6),
                "drift_detected": drift_detected,
                "status": "drift"
                if psi >= 0.25
                else "warning"
                if psi >= 0.1
                else "stable",
            }

        return {
            "drift_detected": global_drift_detected,
            "drifted_features_count": significant_features_count,
            "total_features_count": len(features),
            "metrics": drift_report,
        }
