from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostClassifier

def get_candidate_models(random_state: int):
    """
    Returns dictionary of candidate models for comparison.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            #max_depth=10,
            random_state=random_state,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=random_state
        ),
# Only for Binary Classification Model
        "CatBoost": CatBoostClassifier(
          iterations=200,
          learning_rate=0.05,
          depth=6,
          loss_function="Logloss",
          eval_metric="F1",
          auto_class_weights="Balanced",
           random_seed=random_state,
         verbose=False
        )
    }

def get_candidate_models_ForFarePrediction(random_state: int):
    """
    Returns dictionary of candidate models for comparison.
    """
    return {
        "LinearRegression": LinearRegression(),
          
        "RandomForestRegression": RandomForestRegressor(
            n_estimators=200,
            random_state=random_state
        )
        
    }