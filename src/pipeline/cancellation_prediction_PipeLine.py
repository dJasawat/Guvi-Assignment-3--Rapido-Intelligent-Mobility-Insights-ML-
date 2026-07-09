from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger
import pandas as pd
import numpy as np

logger = setup_logger()

def CustomerCancelPredictionML_Pipeine():
    """
    Creates preprocessing pipeline for numerical and categorical features.
    """
    logger.info("Customer Cancel Prediction - Creating preprocessing pipeline")
   
    numeric_features = [# Original
                            "base_fare",
                            "ride_distance_km",
                            "estimated_ride_time_min",
                            "surge_multiplier",
                            "booking_value",
                            "customer_signup_days_ago",
                           # "total_bookings",
                           # "completed_rides",
                           # "cancelled_rides",
                           # "incomplete_rides",
                           # "cancellation_rate",
                            "avg_customer_rating",
                           
                            # Engineered
                           #"customer_loyalty",
                           #"customer_risk_score"
                           ]

    categorical_features = [ # Original
                            "city",
                            "day_of_week",
                            "traffic_level",
                            "weather_condition",
                            "vehicle_type",
                            "pickup_location",
                            "drop_location",
                            "preferred_vehicle_type",

                            # Engineered
                            "customer_experience",
                            "Weather_Traffic"
                            ]   
    
    binary_features = ["is_weekend","is_night","heavy_traffic",
                       "high_surge","high_fare","is_peak_hour",
                       "is_night_booking",]

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

def customerCancelPrediction_feature_Engineering(df):

    df["is_night"] = ((df["hour_of_day"] >= 22) |
                        (df["hour_of_day"] <= 5)).astype(int)
    
    df["heavy_traffic"] = (df["traffic_level"] == "High").astype(int)
    
    df["high_surge"] = (df["surge_multiplier"] > 1.5).astype(int)

    df["high_fare"] = (df["booking_value"] >df["booking_value"].median()).astype(int)

    df["is_peak_hour"] = df["hour_of_day"].isin([7,8,9,17,18,19]).astype(int)

    df["is_night_booking"] = ((df["hour_of_day"] >= 22) |(df["hour_of_day"] <= 5)).astype(int)
   
    df["customer_loyalty"] = (df["completed_rides"] /(df["total_bookings"]+1))

   

    df["customer_risk_score"] = (df["cancellation_rate"] * 0.7 +(1 - df["customer_loyalty"]) * 0.3)

    df["customer_experience"] = pd.cut(df["total_bookings"],bins=[0,5,20,50,500],labels=["New","Regular","Frequent","VIP"])

  

    # weather_traffic 
    df["Weather_Traffic"] = (df["weather_condition"] +
                                 "_" +
                                df["traffic_level"])
    
    return df

