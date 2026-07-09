from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger

logger = setup_logger()

def split_data(df,target_col,test_size,random_state):
    """
    Splits dataset into train and test sets.
    """
    logger.info("Splitting data into train and test sets")

    x= df.drop(columns=[target_col])
    y=df[target_col]

    x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                     test_size=test_size,
                                                     random_state=random_state,
                                                     )
    
    logger.info(f"Split completed: Train={x_train.shape}, Test={x_test.shape}")


    return x_train, x_test, y_train, y_test