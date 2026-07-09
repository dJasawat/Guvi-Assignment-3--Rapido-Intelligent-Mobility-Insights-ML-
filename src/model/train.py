from sklearn.pipeline import Pipeline
from src.model.model_registry import get_candidate_models,get_candidate_models_ForFarePrediction
from src.model.evaluate import evalute_and_save,evaluate_and_save_fare_prediction
from src.visualization.plot import plot_confusion_matrix
from src.utils.logger import setup_logger
import joblib
import os
import pandas as pd


logger = setup_logger()


def train_and_select_best_model(
    X_train, y_train,
    X_test, y_test,
    preprocessor,
    config,
    ModelFor,
):
    models = get_candidate_models(config["training"]["random_state"])


    best_model = None
    best_score = -1
    best_model_name = None
    best_predictions = None


    for model_name, model in models.items():
        logger.info(f"Training model: {model_name}")


        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])


        pipeline.fit(X_train, y_train)
    #Old Way
     #   y_pred = pipeline.predict(X_test)


      #  _, f1 = evalute_and_save(y_test, y_pred, model_name,ModelFor)
#Old Way end
  #New Way\
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_metrics, test_metrics = evalute_and_save(y_train,y_train_pred,
                                                       y_test,y_test_pred,
                                                       model_name,ModelFor)

        f1 = test_metrics["F1 Score"]   # Use test F1 to select the best model
        if f1 > best_score:
            best_score = f1
            best_model = pipeline
            best_model_name = model_name
            #best_predictions = y_pred
       # checkFeatureImportance(pipeline)

    logger.info(f"Best Model Selected: {best_model_name}")


    # Save best model
    os.makedirs("artifacts/models", exist_ok=True)
    fileName = "artifacts/models/"+ ModelFor +"_bestModel.pkl"
    #joblib.dump(best_model, "artifacts/models/BookingStatusPrediction_best_model.pkl")
    joblib.dump(best_model, fileName)
     
    

    # Save confusion matrix for best model
  #  plot_confusion_matrix(
   #     y_test,
   #    best_predictions,
   #   labels=sorted(y_test.unique()),
   #   save_path="artifacts/reports/confusion_matrix.png"
   #  )


    return best_model_name, best_score

def checkFeatureImportance(pipeline):

    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        print(importance_df.head(20))

    elif hasattr(model, "coef_"):

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": model.coef_[0]
        })

        importance_df["Abs_Coefficient"] = importance_df["Coefficient"].abs()

        importance_df = importance_df.sort_values(
            by="Abs_Coefficient",
            ascending=False
        )

        logger.info(importance_df.head(20))

    else:
        print(f"{type(model).__name__} does not support feature importance.")

def train_and_select_best_model_ForFarePrediction(
    X_train, y_train,
    X_test, y_test,
    preprocessor,
    config
):
    models = get_candidate_models_ForFarePrediction(config["training"]["random_state"])


    best_model = None
    best_score = -1
    best_model_name = None
    best_predictions = None


    for model_name, model in models.items():
        logger.info(f"Training model: {model_name}")


        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])


        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        metrics = evaluate_and_save_fare_prediction(train_y=y_train,
                                                    train_pred=train_pred,
                                                    test_y=y_test,
                                                    test_pred=test_pred,
                                                    model_name=model_name,
                                                    model_for="FarePrediction"
)
       # y_pred = pipeline.predict(X_test)
        


       # metric = evaluate_and_save_fare_prediction(y_test, y_pred, model_name)


        if metrics["R2"] > best_score:
            best_score = metrics["R2"]
            best_model = pipeline
            best_model_name = model_name
           


    logger.info(f"Best Model Selected: {best_model_name}")


    # Save best model
    os.makedirs("artifacts/models", exist_ok=True)
    joblib.dump(best_model, "artifacts/models/FarePrediction_best_model.pkl")

    return best_model_name, best_score

