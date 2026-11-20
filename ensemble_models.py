"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """
    Обучите выбранную модель и верните вероятности, метки и метрики на test.
    """
    
    X_train = data.train_features
    y_train = data.train_target
    X_test = data.test_features
    y_test = data.test_target
    random_seed = data.random_seed

    model = GradientBoostingClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=3, 
        random_state=random_seed
    )
    
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    
    predictions = model.predict(X_test)

    roc_auc = roc_auc_score(y_test, probabilities)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    return EnsembleResult(
        model_name="GradientBoostingClassifier",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=roc_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1
    )