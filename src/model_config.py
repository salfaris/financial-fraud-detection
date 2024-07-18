import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

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
    "svc_rbf": lambda cw: SVC(
        class_weight=cw,
        kernel="rbf",  # Default kernel but want to emphasize.
        tol=1e-3,
        cache_size=1000,
        random_state=RNG,
    ),
}
