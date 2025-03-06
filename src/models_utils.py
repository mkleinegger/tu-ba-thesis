from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from data_utils import split_data
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def create_model(clf, categorical_features):
    """
    Create a model pipeline with one-hot encoding for categorical features.
    """
    categorical_features_onehot_transformer = ColumnTransformer(
        transformers=[
            (
                "one-hot-encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        [
            ("one-hot-encoder", categorical_features_onehot_transformer),
            ("clf", clf),
        ]
    )
    return model


def train_model(model, df, target, drop_na=False, sample_weight=None):
    """
    Train the model and evaluate its performance on a test set.
    """
    X_train, y_train = split_data(df, target, drop_na)

    if sample_weight is not None:
        model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

def calculate_utility_metrics(y_true, y_pred, verbose=False):
    """
    Calculate and print basic performance metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "F1": f1_score(y_true, y_pred, average="macro"),
    }
    if verbose:
        print(f"{'Metric':15} : {'Value':15}")
        for k, v in metrics.items():
            print(f"{k:15} : {v:.3f}")

    return metrics

def evaluate_model(model, X, y, verbose=False):
    """
    Print and return basic performance metrics.
    """
    metrics = calculate_utility_metrics(y,  model.predict(X), verbose)
    return metrics


def train_and_evaluate_pipeline(
    clf, nominal_features, df_train, df_test, target, drop_na=True, verbose=False
):
    """
    Train and evaluate a model pipeline with one-hot encoding for categorical features.
    """
    model = create_model(clf, nominal_features)
    train_model(model, df_train, target, drop_na=drop_na)
    metrics = evaluate_model(model, df_test.drop(columns=[target]), df_test[target], verbose)
    return model, metrics