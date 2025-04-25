import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import sqlite3
import os
from dotenv import load_dotenv
load_dotenv()


DB_PATH = os.getenv("DB_PATH")


def create_synthetic_metric_data():
    np.random.seed(42)
    #number of samples
    n_samples = 10000
    anomaly_fraction = 0.15
    n_anomalies = int(n_samples * anomaly_fraction)
    n_normal = n_samples-n_anomalies

    # Generate normal records
    normal_data = pd.DataFrame({
        'cpu_usage': np.round(np.random.uniform(10, 80, n_normal), 2),
        'memory_usage': [f"{np.random.randint(4, 32)}GB" for _ in range(n_normal)],
        'disk_read_iops': np.random.randint(50, 500, n_normal),
        'disk_write_iops': np.random.randint(50, 500, n_normal),
        'network_in': [f"{np.random.randint(5, 50)}GB" for _ in range(n_normal)],
        'network_out': [f"{np.random.randint(5, 50)}GB" for _ in range(n_normal)],
        'disk_read_latency': np.round(np.random.uniform(1, 5, n_normal), 2),
        'disk_write_latency': np.round(np.random.uniform(1, 5, n_normal), 2),
        'disk_free_space': [f"{np.random.randint(100, 800)}GB" for _ in range(n_normal)],
        'packet_loss': np.round(np.random.uniform(0, 1, n_normal), 2),
        'network_latency': np.round(np.random.uniform(10, 100, n_normal), 2),
        'query_execution_time': np.round(np.random.uniform(10, 200, n_normal), 2),
        'active_connections': np.random.randint(10, 500, n_normal),
        'http_requests_per_sec': np.random.randint(100, 2000, n_normal),
        'backend_latency': np.round(np.random.uniform(50, 400, n_normal), 2),
        'label': 1
    })

    n_obvious_anomalies = n_anomalies // 2

    obvious_anomalies = pd.DataFrame({
        'cpu_usage': np.round(np.random.uniform(90, 120, n_obvious_anomalies), 2),
        'memory_usage': [f"{np.random.randint(64, 256)}GB" for _ in range(n_obvious_anomalies)],
        'disk_read_iops': np.random.randint(2000, 5000, n_obvious_anomalies),
        'disk_write_iops': np.random.randint(2000, 5000, n_obvious_anomalies),
        'network_in': [f"{np.random.randint(100, 500)}GB" for _ in range(n_obvious_anomalies)],
        'network_out': [f"{np.random.randint(100, 500)}GB" for _ in range(n_obvious_anomalies)],
        'disk_read_latency': np.round(np.random.uniform(20, 100, n_obvious_anomalies), 2),
        'disk_write_latency': np.round(np.random.uniform(20, 100, n_obvious_anomalies), 2),
        'disk_free_space': [f"{np.random.randint(1, 10)}GB" for _ in range(n_obvious_anomalies)],
        'packet_loss': np.round(np.random.uniform(10, 50, n_obvious_anomalies), 2),
        'network_latency': np.round(np.random.uniform(500, 2000, n_obvious_anomalies), 2),
        'query_execution_time': np.round(np.random.uniform(800, 3000, n_obvious_anomalies), 2),
        'active_connections': np.random.randint(2000, 5000, n_obvious_anomalies),
        'http_requests_per_sec': np.random.randint(15000, 30000, n_obvious_anomalies),
        'backend_latency': np.round(np.random.uniform(2000, 5000, n_obvious_anomalies), 2),
        'label': -1
    })

    n_subtle = n_anomalies -n_obvious_anomalies

    subtle_anomalies = pd.DataFrame({
        'cpu_usage': np.round(np.random.uniform(80, 95, n_subtle), 2),
        'memory_usage': [f"{np.random.randint(40, 64)}GB" for _ in range(n_subtle)],
        'disk_read_iops': np.random.randint(1000, 2000, n_subtle),
        'disk_write_iops': np.random.randint(1000, 2000, n_subtle),
        'network_in': [f"{np.random.randint(60, 100)}GB" for _ in range(n_subtle)],
        'network_out': [f"{np.random.randint(60, 100)}GB" for _ in range(n_subtle)],
        'disk_read_latency': np.round(np.random.uniform(10, 20, n_subtle), 2),
        'disk_write_latency': np.round(np.random.uniform(10, 20, n_subtle), 2),
        'disk_free_space': [f"{np.random.randint(20, 100)}GB" for _ in range(n_subtle)],
        'packet_loss': np.round(np.random.uniform(1, 5, n_subtle), 2),
        'network_latency': np.round(np.random.uniform(150, 500, n_subtle), 2),
        'query_execution_time': np.round(np.random.uniform(300, 800, n_subtle), 2),
        'active_connections': np.random.randint(800, 2000, n_subtle),
        'http_requests_per_sec': np.random.randint(5000, 15000, n_subtle),
        'backend_latency': np.round(np.random.uniform(800, 2000, n_subtle), 2),
        'label': -1
    })

    df = pd.concat([normal_data,obvious_anomalies,subtle_anomalies], axis =0).sample(frac =1.0, random_state=42).reset_index(drop = True)

    for i in range(len(df)):
        df.loc[i, 'cpu_usage'] += (i / len(df)) * 5  # small increasing trend

    df['network_latency'] = df['packet_loss'] * np.random.uniform(10, 30) + np.random.normal(0, 5, len(df))
    df['backend_latency'] = df['query_execution_time'] * np.random.uniform(1.5, 2.0)

    flip_count = int(0.02 * len(df))
    flip_indices = np.random.choice(df.index, size=flip_count, replace=False)
    df.loc[flip_indices, 'label'] *= -1

    #timestamp column (15 second intervals)
    start_time = datetime.now()
    timestamps = [start_time + timedelta(seconds = 15*i) for i in range(n_samples)]
    df.insert(0,'timestamp',timestamps)


    # #save to CSV
    # df.to_csv("data/synthetic_metric_data.csv", index =False)
    # print("Synthetic data generated and saved in the form of csv")

    #store it in sqlite db
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("metric_data", conn, if_exists="replace", index =False)
    conn.close()



def main():
    create_synthetic_metric_data()

if __name__ == "__main__":
    main()




