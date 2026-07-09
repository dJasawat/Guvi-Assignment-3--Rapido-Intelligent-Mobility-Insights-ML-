from src.utils.logger import setup_logger
from src.data.load_data import load_raw_data
from src.data.splitData import split_data
from src.model.train import train_and_select_best_model
from src.pipeline.BookingStatusPrediction_PipeLIne import BookingStatusPredictionML_Pipeine,BookingStatusPrediction_feature_Engineering

import yaml

def create_BookingStatus_ClassificationModel():

    try:  
        logger = setup_logger()

        logger.info("Booking Status Prediction Model - ML Pipeline Started")

        with open("config/config.yaml", "r") as file:
            config= yaml.safe_load(file)
    
    
        #File Path 
        file_path = config["data"]["processed_path"] + '\DataForBookingStatusPrediction.csv'
        #load data
        df= load_raw_data(file_path)

        df= BookingStatusPrediction_feature_Engineering(df)
        #split data
        x_train,x_text,y_train,y_test = split_data(df,
                                               target_col="booking_status",
                                               test_size=config["training"]["test_size"],
                                               random_state=config["training"]["random_state"])
    

        #preprocessing
        preprocessor = BookingStatusPredictionML_Pipeine()

        # Train & select best model
        best_model_name, best_score = train_and_select_best_model(
            x_train, y_train,
            x_text, y_test,
            preprocessor,
            config,
            "BookingStatusPrediction"
        )


        logger.info(
            f"Training completed. Best Model: {best_model_name} | F1 Score: {best_score:.4f}"
        )

    except Exception as e:
        logger.info(e)