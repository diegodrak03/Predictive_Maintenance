from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb

from pm import config, data, split


def build_logistic_pipeline(numeric_columns, class_weight=None):
    """
    Build a pipeline that scales the numeric columns, one-hot encodes the
    categorical ones and fits a logistic regression. It is not trained.
    """
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLUMNS),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight=class_weight, max_iter=1000)),
    ])


def build_random_forest_pipeline(numeric_columns, class_weight=None):
    """
    Build a pipeline that does not scale the numeric columns, one-hot encodes the
    categorical ones and fits a random forest classifier. It is not trained.
    """

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLUMNS),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(class_weight=class_weight, n_estimators=300, random_state=config.SEED, n_jobs=-1)),
    ])


def build_xgboost_pipeline(numeric_columns, scale_pos_weight=None):
    """
    Build a pipeline that does not scale the numeric columns, one-hot encodes the
    categorical ones and fits an XGBoost classifier. It is not trained.
    """

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numeric_columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_COLUMNS),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=300, learning_rate=0.1, max_depth=4, random_state=config.SEED, n_jobs=-1)),
    ])

def build_selected_model(numeric_columns=config.SELECTED_NUMERICAL_COLUMNS):
    """
    Build the selected model (decisions.md, section 20): random forest with 300 trees,
    no class weighting, basic + physical numeric columns and Type one-hot encoded.
    The numeric columns can be changed only for the "before the process" comparison.
    """
    return build_random_forest_pipeline(numeric_columns)
