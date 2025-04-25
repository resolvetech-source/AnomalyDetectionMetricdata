import streamlit as st
from PIL import Image
import pandas as pd
from components.data_cleaning import DataCleaning
import joblib
import time
import datetime
from pipeline.llm_pipeline import AnomalyLLMExplainer
from components.model_evaluation import ModelEvaluator

logo = Image.open("src/resolve_tech_solutions_logo.jpg")
model =  joblib.load("src/model/isolation_forest.pkl")
train_df = pd.read_csv("src/data/train_cleaned.csv")

llm_explainer =AnomalyLLMExplainer()
model_evaluator = ModelEvaluator()
st.set_page_config(page_title="ResolveTech Solutions", layout="wide")
# Sidebar layout


with st.sidebar:
    # Logo and company name
    col1, col2 = st.columns([1,3])  # Adjust ratio to control spacing
    with col1:
        st.image(logo, width=75)
    with col2:
        st.markdown("**ResolveTech Solutions**")
    
    # Navigation option
    page = st.radio("Navigation", ["Anomaly Detection"])

if page == "Anomaly Detection":
    st.header("Anomaly Detection for Metric Data (Real time simulation)")
    test_data = pd.read_csv("src/data/unseen_test_metric_data.csv")

    st.write("Click the button below to start real-time anomaly detection")

    if "started" not in st.session_state:
        st.session_state.started = False

    if st.button("Start Detection") and not st.session_state.started:
        st.session_state.running =True
        for i in range(len(test_data)):
            row =  test_data.iloc[[i]]
            data_cleaning = DataCleaning()
            cleaned_row = data_cleaning.clean(row)

            prediction =  model.predict(cleaned_row)[0]
            formatted_time = datetime.datetime.now().strftime('%d-%m-%Y  %H:%M:%S')
            #formatted_time = pd.to_datetime(row['timestamp'].values[0]).strftime('%d-%m-%Y  %H:%M:%S')

            with st.container():
                if prediction == -1:

                    col1, col2 = st.columns(2)
                    with col1:  
                        st.write(f"**Record #{i+1} at {formatted_time}** :heavy_exclamation_mark:")
                    with col2:
                        summary = model_evaluator.generate_anomaly_explaination(cleaned_row, train_df)
                        with st.expander(f" Reasoning (Top 3 anomolous features)", expanded= False):
                            st.write(f"{summary}")
                        
                else:
                    st.write(f"**Record #{i+1} at {formatted_time} ✅**")
                    # col1, col2 = st.columns(2)
                    # with col1:
                    #     st.write(f"**Record #{i+1} at {formatted_time}**")
                    #     st.success("✅")
                    # with col2:
                    #     None
                time.sleep(2)

        
        st.session_state.started = False
        st.success("Real-time anomaly detection completed.")


# with col2:
#     st.subheader("Anomaly Detection for Metric Data")
#     st.subheader("Anamoly Detection")
#     test_data = pd.read_csv("src/data/unseen_test_metric_data.csv")
#     st.title("Real time simulation of Anamoly detection")
#     st.write("Click the button below to start real-time anomaly detection")


#     if "started" not in st.session_state:
#         st.session_state.started = False

#     if st.button("Start Detection") and not st.session_state.started:
#         st.session_state.running =True
#         for i in range(len(test_data)):
#             row =  test_data.iloc[[i]]
#             data_cleaning = DataCleaning()
#             cleaned_row = data_cleaning.clean(row)

#             prediction =  model.predict(cleaned_row)[0]
#             formatted_time = datetime.datetime.now().strftime('%d-%m-%Y  %H:%M:%S')
#             #formatted_time = pd.to_datetime(row['timestamp'].values[0]).strftime('%d-%m-%Y  %H:%M:%S')

#             with st.container():
#                 if prediction == -1:

#                     col1, col2 = st.columns(2)
#                     with col1:  
#                         st.write(f"**Record #{i+1} at {formatted_time}** :heavy_exclamation_mark:")
#                     with col2:
#                         summary = model_evaluator.generate_anomaly_explaination(cleaned_row, train_df)
#                         with st.expander(f"Reasoning ", expanded= False):
#                             st.write(f"{summary}")
                        
#                 else:
#                     st.write(f"**Record #{i+1} at {formatted_time} ✅**")
#                     # col1, col2 = st.columns(2)
#                     # with col1:
#                     #     st.write(f"**Record #{i+1} at {formatted_time}**")
#                     #     st.success("✅")
#                     # with col2:
#                     #     None
#                 time.sleep(2)

        
#         st.session_state.started = False
#         st.success("Real-time anomaly detection completed.")


