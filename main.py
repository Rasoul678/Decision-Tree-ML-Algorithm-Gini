from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from DecisionTree import DecisionTree

# Load dataset
diabetes = load_diabetes()
feature_names = diabetes["feature_names"]
diabetes_df = pd.DataFrame(data=diabetes.data, columns=feature_names)
# print(diabetes_df.head(4))

breast_cancer = load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the tree
tree = DecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Make predictions
predictions = tree.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# depths = [3,5,7,10]
# for depth in depths:
#     tree = DecisionTree(max_depth=depth)
#     tree.fit(X_train, y_train)
#     predictions = tree.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
#     print(f"Depth: {depth}, Accuracy: {accuracy:.2f}")

