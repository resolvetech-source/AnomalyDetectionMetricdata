import pandas as pd
import numpy as np

from sklearn.metrics import classification_report



class ModelEvaluator:
    def __init__(self):
        pass

    def evalute(self, model, X_val:pd.DataFrame, val_true:pd.Series):
        X_val_noisy =X_val.copy()
        moise_level = 0.1
        for col in X_val_noisy.select_dtypes(include=['float','int']).columns:
            X_val_noisy[col] += np.random.normal(0, moise_level, size = X_val_noisy.shape[0])

        val_pred = model.predict(X_val_noisy)
        #val_pred = np.where(-1,1,0)
        return classification_report(val_true, val_pred)

    def generate_anomaly_explaination(self, anomaly_row: pd.Series,train_df: pd.DataFrame, top_k:int =3 ):
        explaination =[]

        numerical_columns = train_df.select_dtypes([np.number]).columns

        for col in numerical_columns:
            train_mean = train_df[col].mean()
            train_std = train_df[col].std()
            value =anomaly_row[col].values[0]

            if train_std == 0:
                continue

            z_score = (value -train_mean)/ train_std
            explaination.append((col, value, train_mean, z_score))

        explaination = sorted(explaination, key = lambda x: abs(x[1]), reverse =True)[:top_k]

        summary_lines = []
        for col, val, mean, z in explaination:
            col = col.replace("_"," ").capitalize()
            summary_lines.append(
                f"- `{col}` has a value of **{val:.2f}**, which deviates from the normal mean of **{mean:.2f}** by **{z:.2f} standard deviations**."
            )

        summary_text = "\n".join(summary_lines)

        prompt = (
            f"{summary_text}\n\n"
            f"can you explain in a simple language about the {summary_text} making it as an anomaly"
        )

        return summary_text






