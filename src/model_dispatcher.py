from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier



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
    "catboost": CatBoostClassifier(verbose=0)
}