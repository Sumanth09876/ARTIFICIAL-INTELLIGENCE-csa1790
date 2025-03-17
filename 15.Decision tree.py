import math

# Function to calculate entropy
def entropy(y):
    """Calculate the entropy of a label distribution."""
    label_counts = {}
    for label in y:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    total = len(y)
    ent = 0.0
    for count in label_counts.values():
        probability = count / total
        ent -= probability * math.log2(probability)
    return ent

# Function to split the dataset
def split_dataset(X, y, feature_index, threshold):
    """Split the dataset based on a feature and a threshold."""
    left_X, left_y = [], []
    right_X, right_y = [], []

    for i, sample in enumerate(X):
        if sample[feature_index] <= threshold:
            left_X.append(sample)
            left_y.append(y[i])
        else:
            right_X.append(sample)
            right_y.append(y[i])
    
    return (left_X, left_y), (right_X, right_y)

# Function to calculate information gain
def information_gain(y, left_y, right_y):
    """Calculate the information gain of a split."""
    total_entropy = entropy(y)
    p_left = len(left_y) / len(y)
    p_right = len(right_y) / len(y)
    gain = total_entropy - (p_left * entropy(left_y) + p_right * entropy(right_y))
    return gain

# Class for a Decision Tree node
class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Class for the Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # If all labels are the same, or max depth is reached, make a leaf node
        if len(set(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return DecisionTreeNode(value=y[0])

        # Find the best split
        best_gain = 0
        best_split = None
        for feature_index in range(len(X[0])):
            thresholds = set(sample[feature_index] for sample in X)
            for threshold in thresholds:
                (left_X, left_y), (right_X, right_y) = split_dataset(X, y, feature_index, threshold)
                if len(left_y) == 0 or len(right_y) == 0:
                    continue
                gain = information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left": (left_X, left_y),
                        "right": (right_X, right_y),
                    }

        if best_split is None:
            return DecisionTreeNode(value=max(set(y), key=y.count))

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(*best_split["left"], depth + 1)
        right_subtree = self._build_tree(*best_split["right"], depth + 1)

        return DecisionTreeNode(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree,
        )

    def predict(self, X):
        return [self._predict_sample(sample, self.root) for sample in X]

    def _predict_sample(self, sample, node):
        if node.value is not None:
            return node.value
        if sample[node.feature_index] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)

# Example usage
if __name__ == "__main__":
    # Sample dataset (Iris-like)
    X = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [5.8, 2.7, 5.1, 1.9],
        [6.0, 2.2, 5.0, 1.5],
        [5.4, 3.9, 1.7, 0.4],
    ]
    y = [0, 0, 1, 1, 0]

    # Train the decision tree
    tree = DecisionTree(max_depth=2)
    tree.fit(X, y)

    # Make predictions
    predictions = tree.predict(X)
    print(f"Predictions: {predictions}")
