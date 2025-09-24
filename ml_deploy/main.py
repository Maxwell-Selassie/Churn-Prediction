from fastapi import FastAPI
from pydantic import BaseModel,RootModel
import pandas as pd
import joblib

# initialize fast api
app = FastAPI(title='Customer Churn Prediction API')

model = joblib.load('ml_deploy/calibrated_model.pkl') # load model
feature_names = model.feature_names_in_

# define input schema
class CustomerData(RootModel):
    pass

@app.post('/predict')
def predict(data: CustomerData):
    # convert input data to dataFrame
    input_df = pd.DataFrame([data.root])

    input_df = input_df.reindex(columns=feature_names,fill_value=0.0)

    # get prediction
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    return {
        'prediction' : prediction,
        'probability' : proba
    }
