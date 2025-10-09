# load the best model
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S')

def load_model(model_path: str = 'models/XGBClassifier_best_model.pkl'):
    try:
        best_model = joblib.load(model_path)
        logging.info('Best model successfully loaded!')
    except FileNotFoundError as e:
        raise FileNotFoundError('Model not found!') from e

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    calibrated_model = CalibratedClassifierCV(best_model,method='sigmoid',cv=cv)

    return calibrated_model, best_model
