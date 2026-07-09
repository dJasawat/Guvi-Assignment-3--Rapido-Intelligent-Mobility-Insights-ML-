from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.utils.logger import setup_logger
import numpy as np

logger = setup_logger()

def BookingStatusPredictionML_Pipeine():
    """
    Creates preprocessing pipeline for numerical and categorical features.
    """
    logger.info("Creating preprocessing pipeline")
   
    numeric_features = [ # Original
                            "hour_of_day","ride_distance_km",
                            "estimated_ride_time_min","base_fare","surge_multiplier",
                            "booking_value","customer_age","customer_signup_days_ago",
                            "cancellation_rate","avg_customer_rating","driver_age",
                            "driver_experience_years",
                            "acceptance_rate",
                            "delay_rate",
                            "avg_driver_rating",
                            "avg_pickup_delay_min",

                            # Engineered
                        "Fare_per_KM","Fare_per_Min",
                        "Driver_Reliability_Score",
                        "Customer_Loyalty_Score"]

    categorical_features = [ # Original
                            "city",
                            "day_of_week",
                            "traffic_level",
                            "weather_condition",
                            "preferred_vehicle_type",
                            "pickup_location",
                            "drop_location",
                             # Engineered
                            "Time_of_Day",
                            "route_Pair",
                            "Weather_Traffic"
                            ]   
    
    binary_features = ["is_weekend","Long_Distance_Flag"]

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

def BookingStatusPrediction_feature_Engineering(df):
   # extracting Fare_per_KM from booking value and ride distance 
    df["Fare_per_KM"] = np.where(
                             df["ride_distance_km"] > 0,
                             df["booking_value"] / df["ride_distance_km"],
                             np.nan)
    
    # extracing "Fare_per_Min from booking value and estimated distance"
    df["Fare_per_Min"] = np.where(df["estimated_ride_time_min"] > 0,
                                  df["booking_value"] / df["estimated_ride_time_min"],
                                  np.nan)
    
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

    # Long_Distance_Flag if distance is greater than 0.75% quantile
    threshold = df["ride_distance_km"].quantile(0.75)
    df["Long_Distance_Flag"] = (df["ride_distance_km"] > threshold).astype(int)

    # Pickup and drop location pair
    df["route_Pair"] = ( df["pickup_location"] +"-" +df["drop_location"])

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
    # customer reliability score
    customer_cols = [
          "customer_signup_days_ago",
          "avg_customer_rating",
          "cancellation_rate"
          ]
    
    scaler = MinMaxScaler()
    df[customer_cols] = scaler.fit_transform(df[customer_cols])

    df["Customer_Loyalty_Score"] = (
                                     0.40 * df["customer_signup_days_ago"]
                                    + 0.35 * df["avg_customer_rating"]
                                    + 0.25 * (1 - df["cancellation_rate"])
                                    )
    # weather_traffic 
    df["Weather_Traffic"] = (df["weather_condition"] +
                                 "_" +
                                df["traffic_level"])
    
    return df

