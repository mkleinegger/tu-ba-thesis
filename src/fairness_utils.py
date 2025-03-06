import numpy as np
import pandas as pd
import fairlens as fl
from matplotlib import pyplot as plt
from aif360.detectors.mdss_detector import bias_scan
from aif360.datasets import StandardDataset
from collections import OrderedDict
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import (
    equal_opportunity_difference,
    average_odds_difference,
    statistical_parity_difference,
    disparate_impact_ratio,
    theil_index,
)
from aif360.algorithms.preprocessing import Reweighing
from models_utils import create_model, evaluate_model, train_model
from data_utils import split_data
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from aif360.sklearn.metrics import make_scorer, statistical_parity_difference


def search_bias(
    data,
    observations,
    expectations,
    favorable_value,
    scoring="Bernoulli",
    penalty=1,
    alpha=0.24,
):
    """
    Searches for bias in the given data and returns the privileged and unprivileged groups.
    """
    privileged_subset = bias_scan(
        data=data,
        observations=observations,
        scoring=scoring,
        expectations=expectations,
        overpredicted=True,
        penalty=penalty,
        alpha=alpha,
        favorable_value=favorable_value,
    )

    unprivileged_subset = bias_scan(
        data=data,
        observations=observations,
        scoring=scoring,
        expectations=expectations,
        overpredicted=False,
        penalty=penalty,
        alpha=alpha,
        favorable_value=favorable_value,
    )
    return privileged_subset, unprivileged_subset


def evaluate_fairness_score(data, protected_attributes, target, verbose=False):
    """
    Evaluates the fairness score of the given data.
    """
    df = data.copy()
    df = df.dropna()
    df = df.reset_index()

    fscorer = fl.FairnessScorer(
        df, target_attr=target, sensitive_attrs=protected_attributes
    )

    if verbose:
        fscorer.demographic_report()

    return fscorer


def explain_bias(data, expectations, target, subset, subset_type="privileged"):
    """
    Explains the bias in the given data based on the subset.
    """
    df = data.copy()
    df["expectations"] = expectations.copy()

    to_choose = df[subset.keys()].isin(subset).all(axis=1)
    to_choose = df.loc[to_choose]

    print(
        "Our detected {} group has a size of {}, we observe {} as the average probability of earning >50k, but our model predicts {}".format(
            subset_type,
            len(to_choose),
            np.round(to_choose[target].mean(), 4),
            np.round(to_choose["expectations"].mean(), 4),
        )
    )


def train_and_evaluate_fairness_pipeline(
    clf,
    nominal_features,
    df_train,
    target,
    privileged_subset,
    drop_na=True,
    verbose=False,
):
    """
    Trains a model and evaluates its fairness.
    """
    model = create_model(clf, nominal_features)
    train_model(model, df_train, target, drop_na=drop_na)
    metrics = evaluate_fairness(
        df_train[target],
        model.predict(df_train.drop(columns=[target])),
        list(privileged_subset[0].keys()),
        verbose=verbose,
    )

    return model, metrics


def encode_protected_attributes(
    data, protected_attributes, priveleged_groups, increase_bias=False, verbose=False
):
    """
    Encodes protected attributes into binary format for bias analysis.
    Returns a DataFrame with encoded protected attributes.
    """
    column_renaming = {}
    df = data.copy()
    df = df.dropna()
    if verbose and len(data) != len(df):
        print(f"{len(data)-len(df)} Na rows removed!")

    for i, col in enumerate(protected_attributes):
        df[f"{col}_index"] = df[col].copy()
        if increase_bias:
            df[f"{col}_index"] = df[f"{col}_index"].map(
                lambda x: 1 if x in priveleged_groups[i] else 0
            )
        df[f"{col}"] = df[col].map(lambda x: 1 if x in priveleged_groups[i] else 0)
        column_renaming[f"{col}_index"] = f"{col}"

    if len(protected_attributes) > 0:
        df = df.set_index(protected_attributes)
        df = df.rename(columns=column_renaming)

    return df


def evaluate_fairness(observations, expectations, protected_attributes, verbose=False):
    """
    Evaluates the fairness of the given data.
    """
    privileged_group = (1,) * len(protected_attributes)
    fairness_metrics = {}

    fairness_metrics["statistical_parity_difference"] = statistical_parity_difference(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["average_odds_difference"] = average_odds_difference(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["equal_opportunity_difference"] = equal_opportunity_difference(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["disparate_impact"] = disparate_impact_ratio(
        observations,
        expectations,
        prot_attr=protected_attributes,
        priv_group=privileged_group,
    )
    fairness_metrics["theil_index"] = theil_index(1 + expectations - observations)

    if verbose:
        print(f"{'Metric':30} : {'Value':15}")
        for k, v in fairness_metrics.items():
            print(f"{k:32}{v:.3f}")

    return fairness_metrics


def search_and_evaluate_fairness(model, data, target, penalty):
    """
    Searches for bias and evaluates the fairness of the given data.
    """
    X_train, y_train = split_data(data, target, drop_na=True)
    model.fit(X_train, y_train)
    y_pred, y_probs = model.predict(X_train), pd.Series(
        model.predict_proba(X_train)[:, 1]
    )

    privileged_subset = bias_scan(
        data=X_train,
        observations=y_train,
        scoring="Bernoulli",
        expectations=y_probs,
        overpredicted=True,
        penalty=penalty,
        alpha=0.24,
        favorable_value=1,
    )

    df_bias = encode_protected_attributes(
        data,
        list(privileged_subset[0].keys()),
        list(privileged_subset[0].values()),
    )
    if len(privileged_subset[0].keys()) == 0:
        return {
            "statistical_parity_difference": 0,
            "average_abs_odds_difference": 0,
            "equal_opportunity_difference": 0,
            "disparate_impact": 1,
            "theil_index": 0,
        }, privileged_subset
    else:
        return (
            evaluate_fairness(
                df_bias[target], y_pred, list(privileged_subset[0].keys())
            ),
            privileged_subset,
        )


def compute_metrics(
    dataset_true, dataset_pred, unprivileged_groups, privileged_groups, disp=True
):
    """
    Computes fairness metrics for the given datasets.
    """
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
    metrics = OrderedDict()
    metrics["Statistical parity difference"] = (
        classified_metric_pred.statistical_parity_difference()
    )
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = (
        classified_metric_pred.average_abs_odds_difference()
    )
    metrics["Equal opportunity difference"] = (
        classified_metric_pred.equal_opportunity_difference()
    )
    metrics["Theil index"] = classified_metric_pred.theil_index()

    if disp:
        for k in metrics:
            print("%s = %.4f" % (k, metrics[k]))

    return metrics


def convert_to_standardDataset(
    data,
    categorical_features,
    target,
    favorable_classes,
    protected_attributes,
    privileged_groups,
):
    """
    Converts the given data into a StandardDataset.
    """
    df = data.copy()

    if isinstance(favorable_classes, int):
        favorable_classes = [favorable_classes]

    privileged_classes = []
    for i, group in enumerate(privileged_groups):
        if protected_attributes[i] not in categorical_features:
            privileged_classes.append(lambda x, group=group: x in group)
        else:
            privileged_classes.append(group)

    dataset = StandardDataset(
        df=df,
        label_name=target,
        favorable_classes=favorable_classes,
        scores_name="",
        protected_attribute_names=protected_attributes,
        privileged_classes=privileged_classes,
        categorical_features=[
            col for col in categorical_features if col not in protected_attributes
        ],
    )

    dataset.scores = dataset.labels.copy()
    return dataset


def reweight_mitigation(
    clf,
    nominal_features,
    target,
    df_train,
    df_test,
    penalty=5,
    sample_weights=None,
):
    """
    Reweights the given data to mitigate bias.
    """
    # split data
    X_train, y_train = split_data(df_train, target, True)
    X_test, y_test = split_data(df_test, target, True)

    # train model
    model = create_model(clf, nominal_features)
    model.fit(X_train, y_train, clf__sample_weight=sample_weights)

    # search for bias
    privileged_subset, _ = search_bias(
        X_train,
        y_train,
        pd.Series(model.predict_proba(X_train)[:, 1]),
        1,
        penalty=penalty,
    )
    if len(privileged_subset[0].keys()) <= 0:  # if bias-free, just return
        return None, None, None

    # evaluate model and bias
    model_metrics = evaluate_model(model, X_test, y_test, verbose=False)
    sd_train = convert_to_standardDataset(
        df_train,
        nominal_features,
        target,
        1,
        list(privileged_subset[0].keys()),
        list(privileged_subset[0].values()),
    )
    if sample_weights is not None:
        sd_train.instance_weights = sample_weights

    privileged_groups = [{key: 1 for key in list(privileged_subset[0].keys())}]
    unprivileged_groups = [{key: 0 for key in list(privileged_subset[0].keys())}]

    df_train_bias = encode_protected_attributes(
        df_train,
        list(privileged_subset[0].keys()),
        list(privileged_subset[0].values()),
    )
    fair_metrics = evaluate_fairness(
        df_train_bias[target],
        model.predict(X_train),
        list(privileged_subset[0].keys()),
        verbose=False,
    )

    # reweighing
    RW = Reweighing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )
    ds_reweigh = RW.fit_transform(sd_train)
    return ds_reweigh.instance_weights, model_metrics, fair_metrics


def get_fair_learning_scoring(protected_attributes):
    """
    Returns a scoring function for fair learning.
    """
    def discrimination(y_true, y_pred, protected_attributes):
        return abs(
            statistical_parity_difference(
                y_true,
                y_pred,
                prot_attr=protected_attributes,
                priv_group=(1,) * len(protected_attributes),
            )
        )

    def delta(y_true, y_pred, protected_attributes=None, use_bal_acc=False):
        if use_bal_acc:
            return balanced_accuracy_score(y_true, y_pred) - discrimination(
                y_true, y_pred, protected_attributes
            )
        else:
            return accuracy_score(y_true, y_pred) - discrimination(
                y_true, y_pred, protected_attributes
            )

    return make_scorer(
        delta, protected_attributes=protected_attributes, use_bal_acc=False
    )
