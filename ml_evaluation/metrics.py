from sklearn.metrics import (
    classification_report,precision_score,recall_score,roc_auc_score, brier_score_loss
    ,average_precision_score,log_loss,matthews_corrcoef,cohen_kappa_score
)

import warnings
warnings.filterwarnings('ignore')
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s', datefmt='%H:%M:%S')

def evaluate_model(y_test,y_pred,y_probs):
    '''Prints main metrics'''
    logging.info(f"Precision score : {round(precision_score(y_test,y_pred), 2) * 100}%")
    logging.info(f'Recall score : {round(recall_score(y_test,y_pred), 2) * 100}%')
    logging.info(f'\nClassification Report \n {classification_report(y_test,y_pred)}')
    logging.info(f'ROC_AUC : {round(roc_auc_score(y_test, y_probs), 2) * 100}%')
    logging.info(f'PR_AUC : {round(average_precision_score(y_test,y_probs), 2) * 100}%')
    logging.info(f'Brier loss : {brier_score_loss(y_test, y_probs):.3f}')
    logging.info(f'Log Loss : {log_loss(y_test,y_pred):.3f}')
    logging.info(f'Matthews Corrcoef: {round(matthews_corrcoef(y_test,y_pred),2 ) * 100}%')
    logging.info(f'Cohen_Kappa score : {round(cohen_kappa_score(y_test,y_pred), 2) * 100}%')