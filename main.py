import os
import json
import logging
from typing import Dict, Any
import pandas as pd
import joblib
from fastapi import FastAPI,HTTPException,status
from fastapi.middleware.cors import CORSMiddleware

log = logging.getLogger('ChurnAPI')
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",datefmt='%H:%M:%S')

model_path = 'models/XGBClassifier_best_model.pkl'
feature_path = 'models/feature_names.json'

app = FastAPI(title='Customer Churn Prediction API', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins = ['*'],
    allow_credentials = True,
    allow_headers = ['*'],
    allow_methods = ['*']
)

@app.on_event('startup')
def load_artifacts():
    # load the model
    if not os.path.exists(model_path):
        log.error('Model Not Found %s', model_path)
        raise FileNotFoundError('File Not Found: ', model_path)
    log.info('Model from %s',model_path)
    model = joblib.load(model_path)
    app.state.model = model

    # load feature names else try model attributes
    if os.path.exists(feature_path):
        log.info('Loading feature names from %s', feature_path)
        with open(feature_path,'r') as file:
            feature_names = json.load(file)
            app.state.feature_names = feature_names

    else:
        log.warning('Feature names JSON not found. Trying to read from models.feature_names_in_')
        try:
            app.state.feature_names = list(model.feature_names_in_)
        except Exception as e:
            log.exception('feature names not found anywhere. Aborting startup')
            raise RuntimeError('Feature names unavailable. Save feature list during training')
    
    log.info('Model data and metadata loaded. Expecting %d features.', len(app.state.feature_names))

@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': hasattr(app.state, 'model')}

@app.post('/predict')
def predict(payload: Dict[str, Any]):
    """
    Accepts a JSON payload of feature_name: value pairs.
    Example:
    {
        "Tenure": 19,
        "PreferredLoginDevice": "Phone",
        "Gender": 'Male'
    }
    """
    # basic validation
    if not hasattr(app.state, 'model'):
        raise HTTPException(status_code=500, detail='Model Not Loaded')
    
    # convert payload to dataframe
    try:
        input_df = pd.DataFrame([payload])
    except Exception as e:
        log.exception('Bad JSON input')
        raise HTTPException(status_code=400, detail=f'Error aligning input features {str(e)}')

    # prediction
    model = app.state.model
    try:
        pred = model.predict(input_df)
        pred_value = int(pred[0]) if hasattr(pred[0], "item") else int(pred[0])
    except Exception as e:
        log.exception('Prediction failed')
        raise HTTPException(status_code=400, detail=f'Prediction error {str(e)}')
    
    # probability (if available)
    try:
        proba = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_df)
            proba_value = float(proba[0][1])
        else:
            proba_value = None
    except Exception as e:
        log.warning('Predict probability failed %s', str(e))
        proba_value = None

    return {
        'prediction' : pred_value,
        'probability' : proba_value
    }

        
