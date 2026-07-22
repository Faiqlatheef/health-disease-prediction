import joblib
import pandas as pd


def test_pipeline_load():

    pipeline = joblib.load("../artifacts/best_pipeline.joblib")

    assert pipeline is not None


def test_prediction():

    pipeline = joblib.load("../artifacts/best_pipeline.joblib")

    df = pd.read_csv("../artifacts/clean_dataset.csv")

    X = df.drop(columns=["Has_Disease"])

    prediction = pipeline.predict(X.iloc[[0]])

    assert prediction[0] in [0, 1]


def test_probability():

    pipeline = joblib.load("../artifacts/best_pipeline.joblib")

    df = pd.read_csv("../artifacts/clean_dataset.csv")

    X = df.drop(columns=["Has_Disease"])

    probability = pipeline.predict_proba(X.iloc[[0]])

    assert probability.shape == (1, 2)