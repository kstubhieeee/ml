import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# 1. Create a synthetic dataset
X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, flip_y=0.05, class_sep=1.5, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Initialize base classifier (weak learner)
base_tree = DecisionTreeClassifier(max_depth=1, random_state=42)

# 3. Bagging Classifier
bagging_clf = BaggingClassifier(
    estimator=base_tree,
    n_estimators=50,
    bootstrap=True,
    random_state=42
)
bagging_clf.fit(X_train, y_train)
y_pred_bagging = bagging_clf.predict(X_test)

# 4. Boosting Classifier (AdaBoost)
boosting_clf = AdaBoostClassifier(
    estimator=base_tree,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
boosting_clf.fit(X_train, y_train)
y_pred_boosting = boosting_clf.predict(X_test)

# 5. Voting Classifier (Ensemble of different models)
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_train, y_train)
voting_clf = VotingClassifier(
    estimators=[('bag', bagging_clf), ('boost', boosting_clf), ('tree', tree_clf)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

# 6. Accuracy Scores
print("Bagging Accuracy :", accuracy_score(y_test, y_pred_bagging))
print("Boosting Accuracy:", accuracy_score(y_test, y_pred_boosting))
print("Voting Accuracy  :", accuracy_score(y_test, y_pred_voting))

# 7. Decision Boundary Plot Function
def plot_decision_boundary(clf, X, y, title):
    # Create a meshgrid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 8. Visualize decision boundaries
plot_decision_boundary(bagging_clf, X_test, y_test, "Bagging Decision Boundary          Kaustubh Bane     4")
plot_decision_boundary(boosting_clf, X_test, y_test, "Boosting (AdaBoost) Decision Boundary          Kaustubh Bane     4")
plot_decision_boundary(voting_clf, X_test, y_test, "Voting Classifier Decision Boundary          Kaustubh Bane     4")
