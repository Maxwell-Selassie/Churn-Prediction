from ml_evaluation.data_loader import load_data
from ml_evaluation.model_loader import load_model
from ml_evaluation.metrics import evaluate_model
from ml_evaluation.visualization import (
    plot_calibration_curve,plot_confusion_matrix,plot_learning_curve,
    plot_precision_recall,plot_precision_recall_curve,plot_roc_curve
)
from ml_evaluation.explanability import permutation_importance, feature_importance, shap
# import warnings
# warnings.filterwarnings('ignore')
import joblib

def main():
    x_train, x_test, y_train, y_test, df = load_data()
    calibrated_model, best_model = load_model()

    calibrated_model.fit(x_train,y_train)
    joblib.dump('../models/calibrated_model.pkl')

    y_probs = calibrated_model.predict_proba(x_test)[:,1]
    threshold = 0.2
    y_pred = (y_probs >= threshold).astype(int)

    evaluate_model(y_test, y_pred, y_probs)
    plot_confusion_matrix(y_test,y_pred)
    plot_learning_curve(best_model, x_train, y_train)
    plot_precision_recall(y_test, y_probs)
    plot_precision_recall_curve(y_test, y_probs)
    plot_roc_curve(y_test, y_probs)
    plot_calibration_curve(y_test, y_probs)

    feature_importance(best_model)
    permutation_importance(best_model, x_test, y_test)
    shap(best_model,x_test)