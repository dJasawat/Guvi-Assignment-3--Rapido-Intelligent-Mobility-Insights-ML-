import streamlit as st
import pandas as pd
import joblib

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from src.pipeline.BookingStatusPrediction_PipeLIne import BookingStatusPrediction_feature_Engineering
from src.pipeline.farePrediction_PipeLine import FarePrediction_feature_Engineering
from src.pipeline.driver_delay_prediction_PipeLine import DriverDelayPrediction_feature_Engineering

st.set_page_config(
    page_title="Rapido ML Dashboard",
    page_icon="🚖",
    layout="wide"
)

options= st.sidebar.radio("Navigate",("Predict Booking Status",
                                      "Fare Forecasting","Customer Cancel Ride Prediction",
                                      "Driver Delay Prediction"))



def FarePrediction():
    try:
        st.subheader("Ride Details")
        c1, c2 = st.columns(2)

        with c1:

            distance = st.number_input(
                "Ride Distance (km)",
                min_value=0.0,
                value=5.0,
                step=0.5
                 )

            ride_time = st.number_input(
              "Ride Time (minutes)",
              min_value=1,
             value=20
                )

            base_fare = st.number_input(
              "Base Fare",
                min_value=1.0,
                 value=80.0
                )

            surge = st.slider(
            "Surge Multiplier",
            1.0,
            5.0,
            1.0
             )

        with c2:

            hour = st.slider(
             "Hour of Day",
             0,
             23,
             10
        )

            traffic = st.selectbox(
             "Traffic Level",
             ["Low", "Medium", "High"]
            )

            weather = st.selectbox(
             "Weather Condition",
                ["Clear", "Rain", "Heavy Rain"]
            )

            vehicle = st.selectbox(
                "Vehicle Type",
             ["Bike", "Auto", "Cab"]
            )

        input_df = pd.DataFrame({"ride_distance_km":[distance],
                                "estimated_ride_time_min":[ride_time],
                                "base_fare":[base_fare],
                                "surge_multiplier":[surge],
                                "hour_of_day":[hour],
                                "traffic_level":[traffic],
                                "weather_condition":[weather],
                                "vehicle_type":[vehicle]}) 
        
        model = joblib.load(r"artifacts\models\FarePrediction_best_model.pkl")
        input_df = FarePrediction_feature_Engineering(input_df)
        if st.button("Predict Fare"):
            prediction = model.predict(
            input_df )
            st.success(prediction[0])

        prob = model.predict(input_df)
        st.write(prob)

    except Exception as e:
        st.exception(e)


def DriverDelayPrediction():
    try:
        st.title("🚗 Driver Delay Prediction")

        st.markdown("Enter ride and driver details to predict whether the driver is likely to cause delays.")

# ==========================================
# Numeric Inputs
# ==========================================
        col1, col2 = st.columns(2)
        with col1:

            hour_of_day = st.slider("Hour of Day",0,23,10)
    

            ride_distance_km = st.number_input("Ride Distance (km)",min_value=0.0,value=8.0)

            estimated_ride_time_min = st.number_input("Estimated Ride Time (min)",min_value=1.0,value=25.0)

            driver_experience_years = st.number_input("Driver Experience (Years)",min_value=0.0,value=5.0)

            total_assigned_rides = st.number_input("Total Assigned Rides",min_value=1,value=1000)

            accepted_rides = st.number_input("Accepted Rides", min_value=0,value=900)

        with col2:

            acceptance_rate = st.slider("Acceptance Rate",0.0,1.0,0.90)

            avg_driver_rating = st.slider("Average Driver Rating",1.0,5.0,4.6)

            avg_pickup_delay_min = st.number_input("Average Pickup Delay (min)",min_value=0.0,value=3.5)

            delay_count = st.number_input("Delay Count",min_value=0,value=20)

            delay_rate = st.slider("Delay Rate",0.0,1.0,0.05)

# ==========================================
# Categorical Inputs
# ==========================================

        st.subheader("Ride Information")

        col3, col4 = st.columns(2)

        with col3:
       
            city = st.selectbox("city",["Bangalore","Mumbai","Delhi","Hyderabad"])
            day_of_week = st.selectbox("Days of Week",["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])

            traffic_level = st.selectbox(
                         "Traffic Level",
                                [
                                   "Low",
                                   "Medium",
                                   "High"
                                ]
    )

        with col4:

            weather_condition = st.selectbox("Weather",[
                                    "Clear",
                                    "Rain",
                                    "Heavy Rain"
                                    ])

            vehicle_type_x = st.selectbox(
                             "Vehicle Type",
                                        [
                                      "Bike",
                                      "Auto",
                                       "Cab"] )

            pickup_location = st.selectbox("Pickup location",["Loc_1","Loc_2","Loc_3","Loc_4","Loc_5","Loc_6","Loc_7","Loc_8","Loc_9","Loc_10"])
    

# ==========================================
# Binary Feature
# ==========================================

        st.subheader("Additional Information")
        is_weekend = st.checkbox("Is Weekend")

     
# ==========================================
# Prediction
# ==========================================

        if st.button("Predict Delay", use_container_width=True):

            input_df = pd.DataFrame({

                                 "hour_of_day": [hour_of_day],
                                 "ride_distance_km": [ride_distance_km],
                                 "estimated_ride_time_min": [estimated_ride_time_min],
                                "driver_experience_years": [driver_experience_years],
                                "total_assigned_rides": [total_assigned_rides],
                                "accepted_rides": [accepted_rides],
                                "acceptance_rate": [acceptance_rate],
                                "avg_driver_rating": [avg_driver_rating],
                                "avg_pickup_delay_min": [avg_pickup_delay_min],
                                "delay_count": [delay_count],
                                "delay_rate": [delay_rate],

                                 "city": [city],
                                 "day_of_week": [day_of_week],
                                 "traffic_level": [traffic_level],
                                 "weather_condition": [weather_condition],
                                 "vehicle_type_x": [vehicle_type_x],
                                 "pickup_location": [pickup_location],

                                 "is_weekend": [int(is_weekend)]
                         })
    
            model = joblib.load(r"artifacts\models\DriverDelayPredictionModel_bestModel.pkl")
            input_df = DriverDelayPrediction_feature_Engineering(input_df)
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            st.divider()

            if prediction == 1:
                 st.error(f"⚠️ Driver is likely to cause delays.\n\nProbability: **{probability:.2%}**")
            else:
                st.success(f"✅ Driver is unlikely to cause delays.\n\nProbability: **{probability:.2%}**")

            with st.expander("Input Data"):
                st.dataframe(input_df, use_container_width=True)
        
        
    except Exception as e:
        st.exception(e)

def BookingStatusPrediction():
    try:
        st.title("Rapido Booking Prediction")

        #Numeric Values

        hour_of_day = st.number_input("Hour of the Day")

        ride_distance_km = st.number_input( "Ride Distance (KM)")

        estimated_ride_time_min = st.number_input("Estimated Ride Time")

        base_fare = st.number_input("Base Fare")
        surge_multiplier = st.number_input("Surge multiplier",min_value=1,max_value=5)
     
        booking_value = st.number_input("Booking Value")

        customer_age = st.number_input("Customer Age")

        customer_signup_days_ago = st.number_input("customer signup days ago")
        cancellation_rate  = st.number_input("Cancellation Rate",min_value=0,max_value=1)
        avg_customer_rating =st.number_input("Average customer rating",min_value=1.0,max_value=5.0)
        driver_age = st.number_input("Driver age",min_value=16,max_value=80)
        driver_experience_years = st.number_input("Driver Experience Years")
        acceptance_rate = st.number_input("Acceptence Rate",min_value=0.0,max_value=1.0)
        delay_rate= st.number_input("Delay Rate", min_value=0.0,max_value=1.0)
        avg_driver_rating = st.number_input("Average Driver Rating",min_value=1,max_value=5)
        avg_pickup_delay_min =st.number_input("Average Picup delay minutes")

        #Categorical columns
        city = st.selectbox("city",["Bangalore","Mumbai","Delhi","Hyderabad"])
        day_of_week = st.selectbox("Days of Week",["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])
        preferred_vehicle_type = st.selectbox("Vehicle Type",["Cab","Bike","Auto"])
        pickup_location = st.selectbox("Pickup location",["Loc_1","Loc_2","Loc_3","Loc_4","Loc_5","Loc_6","Loc_7","Loc_8","Loc_9","Loc_10"])
        drop_location = st.selectbox("Drop Location",["Loc_1","Loc_2","Loc_3","Loc_4","Loc_5","Loc_6","Loc_7","Loc_8","Loc_9","Loc_10"])
        traffic_level = st.selectbox("Traffic",[
                                    "Low",
                                    "Medium",
                                    "High"
                                    ])

        weather_condition = st.selectbox("Weather",[
                                    "Clear",
                                    "Rain",
                                    "Heavy Rain"
                                    ])

        #Binary Columns
        is_weekend = st.checkbox("Is Weekend")
        input_df = pd.DataFrame({ 
        #Numerical Columns
            "hour_of_day" :[hour_of_day],
            "ride_distance_km" :[ride_distance_km],
            "estimated_ride_time_min" : [estimated_ride_time_min],
            "base_fare" : [base_fare],
            "surge_multiplier" :[surge_multiplier],
            "booking_value": [booking_value],
            "customer_age": [customer_age],
            "customer_signup_days_ago" : [customer_signup_days_ago],
            "cancellation_rate" :[cancellation_rate],
            "avg_customer_rating": [avg_customer_rating],
            "driver_age":[driver_age],
            "driver_experience_years":[driver_experience_years],
            "acceptance_rate":[acceptance_rate],
            "delay_rate":[delay_rate],
            "avg_driver_rating":[avg_driver_rating],
            "avg_pickup_delay_min":[avg_pickup_delay_min],
        #Categorical Columns
            "city":[city],
            "day_of_week":[day_of_week],
            "preferred_vehicle_type":[preferred_vehicle_type],
            "pickup_location":[pickup_location],
            "drop_location":[drop_location],
            "traffic_level":[traffic_level],
            "weather_condition":[weather_condition],

            "is_weekend":[is_weekend]
        })

        model = joblib.load(r"artifacts\models\BookingStatusPrediction_bestModel.pkl")
        input_df = BookingStatusPrediction_feature_Engineering(input_df)
        if st.button("Predict"):
            prediction = model.predict(
            input_df )
            st.success(prediction[0])

        prob = model.predict_proba(input_df)
        st.write(prob)

    except Exception as e:
        st.exception(e)


try:
    if options =="Predict Booking Status":
        BookingStatusPrediction()
    if options == "Fare Forecasting":
        FarePrediction()
    if options == "Driver Delay Prediction":
        DriverDelayPrediction()

except Exception as e:
    st.exception(e)