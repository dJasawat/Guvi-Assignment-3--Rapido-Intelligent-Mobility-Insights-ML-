from src.model.FarePredictionMLModel import  create_FarePrediction_RegressionModel
from src.model.BookingStatus_PredictionMLModel import  create_BookingStatus_ClassificationModel
from src.model.cancellation_predictionML import create_CustomerCancelPrediction_BinaryClassificationModel
from src.model.driver_delay_predictionML import create_DriverDelay_BinaryClassificationModel

def main():

    #Model To Predict Fare
   # create_FarePrediction_RegressionModel()

    #Model To Predict Booking Status
    #create_BookingStatus_ClassificationModel()

    #Model To predict Customer Cancel Prediction
    create_CustomerCancelPrediction_BinaryClassificationModel()
    #Model to Predict Diver Delay 
    #create_DriverDelay_BinaryClassificationModel()




if __name__ == "__main__":
    main()

