import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH")
print(DB_PATH)
class DataIngestion:
    def __init__(self):
        self.db_path = DB_PATH
        self.table_name = "metric_data"

    def data_ingest(self) -> pd.DataFrame:
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"DB not found: {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            raw_df = pd.read_sql(f"Select * from {self.table_name}",conn)
            return raw_df

