import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger()

#Load Raw Data For ML Model To get Trained 
def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV file.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

