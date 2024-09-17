from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier


# Define base models

base_models = [
    ('xgboost', XGBClassifier()),
    ('lightgbm', LGBMClassifier()),
    ('catboost', CatBoostClassifier(verbose=0)),
]

# Meta model
meta_model = XGBClassifier()

# Define the stacking classifier
stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "svm": SVC(),
    "gradient_boosting": GradientBoostingClassifier(),
    "xgboost": XGBClassifier(),
    "knn": KNeighborsClassifier(),
    "extra_trees": ExtraTreesClassifier(),
    "ada_boost": AdaBoostClassifier(),
    "bagging": BaggingClassifier(),
    "lightgbm": LGBMClassifier(),
    "catboost": CatBoostClassifier(verbose=0),
    "stacking": stacking_model,
    "xgboost_sg1": XGBClassifier(subsample=0.8, n_estimators=300, max_depth=3, learning_rate=0.1, colsample_bytree=1.0),
    "xgboost_sg": XGBClassifier(subsample=0.9, reg_lambda=0.3333333333333333, reg_alpha=0.0, n_estimators=340, min_child_weight=3, max_depth=3, learning_rate=0.12666666666666668, gamma=0.3333333333333333, colsample_bytree=0.9),
    "xgboost_sg_f": XGBClassifier(subsample=0.7, reg_lambda=0.333, reg_alpha=0.222, n_estimators=330, min_child_weight=5, max_depth=3, learning_rate=0.127, gamma=0.333, colsample_bytree=0.9),

}
