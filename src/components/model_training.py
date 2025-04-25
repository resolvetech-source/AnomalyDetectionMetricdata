from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import joblib

class ModelTrainer:
    def __init__(self):
        self.model = IsolationForest(n_estimators= 100, contamination=0.15,random_state = 42)


    def train(self,X_train:pd.DataFrame):
        self.model.fit(X_train)
        return self.model

    def save_model(self, model, path = "src/model/isolation_forest.pkl"):
        joblib.dump(model, path)

    def save_train_set(self, train_df: pd.DataFrame, path = "src/data/train_cleaned.csv"):
        train_df.to_csv(path)

