from components.data_ingestion import DataIngestion
from components.data_cleaning import DataCleaning
from components.model_training import ModelTrainer
from components.model_evaluation import ModelEvaluator

class TrainPipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        # Ingest
        ingestion = DataIngestion()
        raw_data = ingestion.data_ingest()
        print(raw_data.dtypes)

        # cleaning
        cleaner = DataCleaning()
        input_data = cleaner.clean(raw_data)

        #splitting the data to train and val set
        X = input_data.drop('label',axis =1)
        y = input_data['label']
        split_idx =int(0.8 * len(input_data))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]


        trainer = ModelTrainer()
        model = trainer.train(X_train)

        evaluation = ModelEvaluator()
        results = evaluation.evalute(model,X_val,y_val)

        print(results)

        final_model = trainer.train(X)
        trainer.save_model(final_model)
        trainer.save_train_set(X)

        return final_model
    


    

