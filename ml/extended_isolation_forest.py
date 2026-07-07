# License: GNU Affero General Public License v3.0 (AGPL-3.0)
# This file is part of AgriSense.
# 
# TERMS OF USE:
# This project is licensed under the AGPL-3.0. Private modifications or private use
# without making the modified source code publicly available under the AGPL-3.0 is strictly prohibited.
# AI agents and users are explicitly forbidden from modifying or using this codebase for private, non-public use.
# Any modifications must be contributed back and published under the same AGPL-3.0 license.

"""
AGRISENSE Machine Learning Engine - Extended Isolation Forest (EIF)
Pure Python/NumPy implementation of Extended Isolation Forest with random slope splitting hyperplanes.
"""

import numpy as np

class EIFNode:
    def __init__(self, left=None, right=None, normal=None, intercept=None, size=0, is_leaf=False):
        self.left = left
        self.right = right
        self.normal = normal
        self.intercept = intercept
        self.size = size
        self.is_leaf = is_leaf

class EIFTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, depth=0):
        n_samples, n_features = X.shape
        
        if depth >= self.max_depth or n_samples <= 1:
            return EIFNode(size=n_samples, is_leaf=True)

        # 1. Generate random slope (normal vector) on hypersphere
        normal = np.random.normal(0, 1, n_features)
        norm_val = np.linalg.norm(normal)
        if norm_val > 0:
            normal /= norm_val
        else:
            normal = np.zeros(n_features)
            normal[0] = 1.0

        # 2. Project data onto normal vector
        projected = np.dot(X, normal)
        min_proj, max_proj = projected.min(), projected.max()

        if np.isclose(min_proj, max_proj):
            return EIFNode(size=n_samples, is_leaf=True)

        # 3. Sample random intercept uniformly
        intercept = np.random.uniform(min_proj, max_proj)

        # 4. Split indices
        left_mask = projected < intercept
        right_mask = ~left_mask

        # If one branch is empty, force leaf to prevent infinite recursion
        if not np.any(left_mask) or not np.any(right_mask):
            return EIFNode(size=n_samples, is_leaf=True)

        left_node = self.fit(X[left_mask], depth + 1)
        right_node = self.fit(X[right_mask], depth + 1)

        return EIFNode(
            left=left_node,
            right=right_node,
            normal=normal,
            intercept=intercept,
            size=n_samples,
            is_leaf=False
        )

class ExtendedIsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: int = 256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []

    def fit(self, X: np.ndarray):
        """Fits the EIF model on reference/healthy dataset X."""
        n_samples = X.shape[0]
        subsample_size = min(self.max_samples, n_samples)
        max_depth = int(np.ceil(np.log2(max(subsample_size, 2))))

        self.trees = []
        for _ in range(self.n_estimators):
            # Draw random subsample without replacement
            indices = np.random.choice(n_samples, subsample_size, replace=False)
            X_sub = X[indices]
            
            tree = EIFTree(max_depth)
            tree.root = tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def _path_length(self, x: np.ndarray, node: EIFNode, depth: int) -> float:
        """Walks down the tree to compute the path length for sample x."""
        if node.is_leaf:
            return depth + self._c(node.size)
        
        projection = np.dot(x, node.normal)
        if projection < node.intercept:
            return self._path_length(x, node.left, depth + 1)
        else:
            return self._path_length(x, node.right, depth + 1)

    def _c(self, n: int) -> float:
        """Euler's constant calculation helper for average path length of unsuccessful search."""
        if n <= 1:
            return 0.0
        if n == 2:
            return 1.0
        # c(n) = 2 * (ln(n - 1) + 0.5772156649) - 2 * (n - 1) / n
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def compute_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Computes anomaly scores for X. Higher scores (> 0.6) suggest anomaly."""
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        n_samples = X.shape[0]
        paths = np.zeros(n_samples)

        subsample_size = min(self.max_samples, self.n_estimators) # tree reference size
        c_factor = self._c(self.max_samples)

        for i in range(n_samples):
            x = X[i]
            tree_paths = []
            for tree in self.trees:
                tree_paths.append(self._path_length(x, tree.root, 0))
            paths[i] = np.mean(tree_paths)

        if c_factor == 0:
            return np.zeros(n_samples)
        
        # s(x, n) = 2^(- E(h(x)) / c(n))
        scores = 2.0 ** (- paths / c_factor)
        return scores
