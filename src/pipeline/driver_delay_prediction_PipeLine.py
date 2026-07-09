from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger
import numpy as np

logger = setup_logger()

def DriverDelayPredictionML_Pipeine():
    """
    Creates preprocessing pipeline for numerical and categorical features.
    """
    logger.info("Driver Delay - Creating preprocessing pipeline")
   
    numeric_features = [# Original
                            "hour_of_day","ride_distance_km",
                            "estimated_ride_time_min",
                            "driver_experience_years",
                            "total_assigned_rides",
                            "accepted_rides",
                            "acceptance_rate",
                            "avg_driver_rating",
                            "avg_pickup_delay_min",
                            "delay_count",
                            "delay_rate",
                            

                            # Engineered
                            "driver_utilization",
                            "Driver_Reliability_Score"]

    categorical_features = [ # Original
                            "city",
                            "day_of_week",
                            "traffic_level",
                            "weather_condition",
                            "vehicle_type_x",
                            "pickup_location",
                            # Engineered
                            "Time_of_Day",
                            "Weather_Traffic"
                            ]   
    
    binary_features = ["is_weekend","is_night","heavy_traffic"]

    binary_pipeline= Pipeline(steps=[("passthrough", "passthrough")])    

    numeric_pipeline = Pipeline(steps=[("scaler",StandardScaler())])

    categorical_pipeline =Pipeline(steps=[("encoder" , OneHotEncoder(handle_unknown="ignore"))])

    preprocessor =  ColumnTransformer(
                     transformers=[
                        ("num",numeric_pipeline,numeric_features),
                        ("cat",categorical_pipeline,categorical_features),
                        ("bin",binary_pipeline,binary_features)
                        ])

    logger.info("Preprocessing pipeline created successfully")
    return preprocessor

def DriverDelayPrediction_feature_Engineering(df):

    df["is_night"] = ((df["hour_of_day"] >= 22) |
                        (df["hour_of_day"] <= 5)).astype(int)
    
    df["heavy_traffic"] = (df["traffic_level"] == "High").astype(int)
    
    df["driver_utilization"] = (df["accepted_rides"] /df["total_assigned_rides"])


    
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


    # Driver_Reliability_Score
    driver_cols = [
        "acceptance_rate",
        "avg_driver_rating",
        "driver_experience_years",
        "delay_rate",
        "avg_pickup_delay_min"
    ]
    scaler = MinMaxScaler()

    df[driver_cols] = scaler.fit_transform(df[driver_cols])

    df["Driver_Reliability_Score"] = (
                 0.35 * df["acceptance_rate"]
                 + 0.30 * df["avg_driver_rating"]
                 + 0.20 * df["driver_experience_years"]
                 + 0.10 * (1 - df["delay_rate"])
                 + 0.05 * (1 - df["avg_pickup_delay_min"])
                )

    
  

    # weather_traffic 
    df["Weather_Traffic"] = (df["weather_condition"] +
                                 "_" +
                                df["traffic_level"])
    
    return df

