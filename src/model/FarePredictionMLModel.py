from src.utils.logger import setup_logger
from src.data.load_data import load_raw_data
from src.data.splitData import split_data
from src.model.train import train_and_select_best_model_ForFarePrediction
from src.pipeline.farePrediction_PipeLine import FarePredictionML_Pipeline,FarePrediction_feature_Engineering
import yaml
   
def create_FarePrediction_RegressionModel():

    logger = setup_logger()

    logger.info("Fare Prediction - LinearRegrsession_ML Pipeline Started")

    with open("config/config.yaml", "r") as file:
         config= yaml.safe_load(file)
    
    #File Path 
    file_path = config["data"]["processed_path"] + '\DataForFarePrediction.csv'
    #load data
    df= load_raw_data(file_path)

    logger.info("Step 2: Feature Engineering")
    df= FarePrediction_feature_Engineering(df)

    logger.info("Step 3: Splitting data")
    #split data
    x_train,x_test,y_train,y_test = split_data(df,
                                               target_col="booking_value",
                                               test_size=config["training"]["test_size"],
                                               random_state=config["training"]["random_state"])
    

    #preprocessing

    preprocessor = FarePredictionML_Pipeline()

    # Train & select best model
    best_model_name, best_score = train_and_select_best_model_ForFarePrediction (
        x_train, y_train,
        x_test, y_test,
        preprocessor,
        config
    )


    logger.info(
        f"Training completed. Best Model: {best_model_name} | F1 Score: {best_score:.4f}"
    )
