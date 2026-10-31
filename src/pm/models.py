from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
