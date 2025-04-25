
import pandas as pd

class DataCleaning:
    def __init__(self):
        pass

    def clean(self, df):
        df = df.copy()
        df["memory_usage"] = df["memory_usage"].str.replace("GB", "").astype(float)
        df["network_in"] = df["network_in"].str.replace("GB", "").astype(float)
        df["network_out"] = df["network_out"].str.replace("GB", "").astype(float)
        df["disk_free_space"] = df["disk_free_space"].str.replace("GB", "").astype(float)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace =True) # setting timestamp as index
        df.dropna(inplace=True)
        return df