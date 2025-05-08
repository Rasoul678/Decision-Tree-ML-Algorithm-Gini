import numpy as np
from collections import Counter

# Step 1: Define the Node structure
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left            # Left subtree (<= threshold)
        self.right = right          # Right subtree (> threshold)
        self.value = value          # Class label if it's a leaf node

    def is_leaf_node(self):
        return self.value is not None

# Step 2: Implement the Decision Tree
class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
            or num_labels == 1
            or num_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y)

        # split the dataset
        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = X[:, best_feature] > best_thresh

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for thresh in thresholds:
                # split the dataset
                left_idxs = y[X[:, feature_index] <= thresh]
                right_idxs = y[X[:, feature_index] > thresh]
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                gain = self._information_gain(y, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_index
                    split_thresh = thresh

        return split_idx, split_thresh

    def _information_gain(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        gain = self._gini(parent) - (
            weight_left * self._gini(left) + weight_right * self._gini(right)
        )
        return gain

    def _gini(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
