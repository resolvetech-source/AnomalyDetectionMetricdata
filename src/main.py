from pipeline.train_pipeline import TrainPipeline


if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    final_model= train_pipeline.run_pipeline()


