import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# I only use `np.random.RandomState` over `np.random.default_rng` because sklearn does
# not support the latter yet.
RNG = np.random.RandomState(10062930)

FEATURE_NAMES = [
    "amount",
    "old_balance_Source",
    "new_balance_Source",
    "old_balance_Destination",
    "new_balance_Destination",
]

TARGET_NAME = "is_fraud"

MODEL_FUNCTIONS = {
    "logreg": lambda cw: LogisticRegression(class_weight=cw, random_state=RNG),
    "svc_linear": lambda cw: LinearSVC(
        class_weight=cw,
        tol=1e-5,
        max_iter=1000,
        dual=False,
        random_state=RNG,
    ),
    "svc_rbf_sampler": None,
    "svc_rbf": lambda cw: SVC(
        class_weight=cw,
        # # Apparently using `sklearn.metrics.pairwise.rbf_kernel` utilizes numpy
        # # and uses all available threads leading to a massive speedup.
        # # This was reported as bug:
        # #   https://github.com/scikit-learn/scikit-learn/issues/21410
        # kernel=sklearn.metrics.pairwise.rbf_kernel,
        kernel="rbf",
        tol=1e-3,
        cache_size=1000,
        random_state=RNG,
    ),
    "decision_tree": lambda cw: DecisionTreeClassifier(
        class_weight=cw,
        max_depth=5,  # Avoid overfitting to training set.
        random_state=RNG,
    ),
}
