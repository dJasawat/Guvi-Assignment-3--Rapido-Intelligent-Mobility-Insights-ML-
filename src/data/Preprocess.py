from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger
import numpy as np

logger = setup_logger()

# Create Pipeline For FarePrediction
def create_Pipeline_For_FarePrediction_LinearRegression():
    """
    Creates preprocessing pipeline for numerical and categorical features - for Fare Prediction ML Model.
    """
    logger.info("Fare- Prediction ML -Creating preprocessing pipeline")
   
    numeric_features = [ # Original
                            "ride_distance_km",
                            "estimated_ride_time_min",
                            "base_fare",
                            "surge_multiplier",
                            "hour_of_day"
                        ]
    categorical_features = [ # Original
                           
                            "traffic_level",
                            "weather_condition",
                            "vehicle_type",
                             # Engineered
                            "Time_of_Day",
                            "Weather_Traffic"
                            ]   
  
    binary_pipeline= Pipeline(steps=[("passthrough", "passthrough")])    

    numeric_pipeline = Pipeline(steps=[("scaler",StandardScaler())])

    categorical_pipeline =Pipeline(steps=[("encoder" , OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
                     transformers=[
                        ("num",numeric_pipeline,numeric_features),
                        ("cat",categorical_pipeline,categorical_features),
                        ])

    logger.info("Preprocessing pipeline created successfully")
    return preprocessor

#Feature Engineering for FarePrediction
def FarePrediction_feature_Engineering(df):
    # extract "Time_of_Day" based on the hour of the day coulmn
    def get_time_of_day(hour):
        if 0 <= hour <= 5:
          return "Late Night"
        elif 6 <= hour <= 11:
            return "Morning"
        elif 12 <= hour <= 16:
            return "Afternoon"
        elif 17 <= hour <= 21:
            return "Evening"
        else:
         return "Night"

    df["Time_of_Day"] = df["hour_of_day"].apply(get_time_of_day)


     # weather_traffic 
    df["Weather_Traffic"] = (df["weather_condition"] +
                             
                                 "_" +
                                df["traffic_level"])
    
    return df


