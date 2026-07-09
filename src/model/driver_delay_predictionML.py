from src.utils.logger import setup_logger
from src.data.load_data import load_raw_data
from src.data.splitData import split_data
from src.model.train import train_and_select_best_model
from src.pipeline.driver_delay_prediction_PipeLine import DriverDelayPrediction_feature_Engineering,DriverDelayPredictionML_Pipeine

import yaml

def create_DriverDelay_BinaryClassificationModel():

    try:  
        logger = setup_logger()

        logger.info("DriverDelay Prediction Model - ML Pipeline Started")

        with open("config/config.yaml", "r") as file:
            config= yaml.safe_load(file)
    
    
        #File Path 
        file_path = config["data"]["processed_path"] + '\DataForDriverDelayPrediction.csv'
        #load data
        df= load_raw_data(file_path)

        df= DriverDelayPrediction_feature_Engineering(df)
        #split data
        x_train,x_text,y_train,y_test = split_data(df,
                                               target_col="driver_delay_ride",
                                               test_size=config["training"]["test_size"],
                                               random_state=config["training"]["random_state"])
    

        #preprocessing
        preprocessor = DriverDelayPredictionML_Pipeine()

        # Train & select best model
        best_model_name, best_score = train_and_select_best_model(
            x_train, y_train,
            x_text, y_test,
            preprocessor,
            config,
            "DriverDelayPredictionModel"
        )


        logger.info(
            f"Training completed. Best Model: {best_model_name} | F1 Score: {best_score:.4f}"
        )

    except Exception as e:
        logger.info(e)