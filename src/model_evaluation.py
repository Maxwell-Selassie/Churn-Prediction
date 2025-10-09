import os
from pathlib import Path

print("Current working directory:", os.getcwd())
print("File absolute path:", Path(__file__).resolve())
print("File directory:", Path(__file__).parent)

from ml_evaluation.data_loader import load_data
from ml_evaluation.model_loader import load_model
from ml_evaluation.metrics import evaluate_model
from ml_evaluation.visualization import (
    plot_calibration_curve,plot_confusion_matrix,plot_learning_curve,
    plot_precision_recall,plot_precision_recall_curve,plot_roc_curve
)
from ml_evaluation.explainability import compute_feature_importance, compute_permutation_importance, compute_shap
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s', datefmt='%H:%M:%S')

def main():
    x_train, x_test, y_train, y_test = load_data()
    best_model = load_model('models/XGBClassifier_best_model.pkl')

    y_probs = best_model.predict_proba(x_test)[:,1]
    logging.info('Calibrated model trained successfully on x_test')
    threshold = 0.3
    logging.info(f'Threshold is {threshold}')
    y_pred = (y_probs >= threshold).astype(int)
    logging.info('Model Predictions')

    evaluate_model(y_test, y_pred, y_probs)
    logging.info('Evaluate_model file successfully runned')
    plot_confusion_matrix(y_test,y_pred)
    plot_learning_curve(best_model, x_train, y_train)
    plot_precision_recall(y_test, y_probs)
    plot_precision_recall_curve(y_test, y_probs)
    plot_roc_curve(y_test, y_probs)
    plot_calibration_curve(y_test, y_probs)
    logging.info('Visualization.py file successfully runned')

    compute_feature_importance(best_model)
    compute_permutation_importance(best_model, x_test, y_test)
    compute_shap(best_model,x_test)
    logging.info('Explanability.py file successfully runned')
main()