import os
from sdmetrics.single_table import (
    CSTest,
    ContinuousKLDivergence,
    DiscreteKLDivergence,
    LogisticDetection,
)


def generate_synthetic_data(synthesizer, df, num_rows=None, fit=False):
    """
    Generate synthetic data using the given synthesizer.
    Fits the synthesizer if needed and caches the model.
    """
    os.makedirs("../data/SDV", exist_ok=True)

    def fit_synthesizer(synth, df):
        synth.fit(df)
        synth.save(filepath=f"../data/SDV/{type(synth).__name__}.pkl")

    try:
        if fit:
            fit_synthesizer(synthesizer, df)
        else:
            synthesizer = synthesizer.load(
                filepath=f"../data/SDV/{type(synthesizer).__name__}.pkl"
            )
    except Exception:
        fit_synthesizer(synthesizer, df)

    if num_rows is None:
        num_rows = len(df)
    return synthesizer.sample(num_rows=num_rows)


def evaluate_fidelity(df, df_synthesized):
    """
    Evaluate the fidelity of synthetic data using various statistical metrics.
    """
    results = {
        "CSTest": CSTest.compute(df, df_synthesized),
        "ContinuousKLDivergence": ContinuousKLDivergence.compute(df, df_synthesized),
        "DiscreteKLDivergence": DiscreteKLDivergence.compute(df, df_synthesized),
        "LogisticDetection": LogisticDetection.compute(df, df_synthesized),
    }
    return results
