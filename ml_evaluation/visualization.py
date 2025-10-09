import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, confusion_matrix,ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import learning_curve,LearningCurveDisplay

# confusion matrix - per-class misclassification
def plot_confusion_matrix(y_test,y_pred):
    from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
    cm = confusion_matrix(y_test,y_pred)
    disp = ConfusionMatrixDisplay(cm,display_labels=np.unique(y_test))
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png',bbox_inches='tight',dpi=300)
    plt.show()

# threshold 
def plot_precision_recall_curve(y_test, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    plt.plot(thresholds, precisions[:-1], label="Precision",marker='x')
    plt.plot(thresholds, recalls[:-1], label="Recall",marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig('plots/confusion_matrix.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_learning_curve(best_model, x_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
    best_model,x_train,y_train,cv = 5, scoring='accuracy',shuffle=True,random_state=42,n_jobs= -1
    )
    plt.figure(figsize=(10,7))
    disp = LearningCurveDisplay(
        train_sizes=train_sizes,
        train_scores=train_scores,
        test_scores=test_scores
    )
    disp.plot()
    plt.title('Learning Curve')
    plt.xlabel('Training set sizes')
    plt.ylabel('Accuracy')
    plt.savefig('plots/confusion_matrix.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_roc_curve(y_test, y_probs):
    # the rate at which model classifiers model well - determines whether model is really learning or just randomly guessing
    fpr,tpr,threshold = roc_curve(
        y_test,y_probs
    )
    auc_score = roc_auc_score(y_test,y_probs)
    plt.plot(fpr,tpr,label=f'AUC : {round(auc_score,2)*100}%')
    plt.plot([0,1],[0,1],linestyle='--',color='blue')
    plt.title('ROC Curve')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.savefig('plots/confusion_matrix.png',bbox_inches='tight',dpi=300)
    plt.show()

# precision recall tradeoff
def plot_precision_recall(y_test, y_probs):
    precision,recall,threshold = precision_recall_curve (
    y_test,y_probs
    )
    pr_auc = average_precision_score(y_test,y_probs)
    plt.plot(recall,precision,label=f'PR AUC: {round(pr_auc,2)*100}%',marker='x')
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('plots/confusion_matrix.png',bbox_inches='tight',dpi=300)
    plt.show()

# compares how well predicted probabilties match real outcomes
def plot_calibration_curve(y_test, y_probs):
    prob_true, prob_pred = calibration_curve(
        y_test,y_probs, n_bins=10, strategy='uniform'
    )
    plt.plot(prob_true, prob_pred, marker= 'x')
    plt.plot([0,1],[0,1],linestyle='--',color='blue')
    plt.title('Calibration Curve')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel('Fraction of Positives')
    plt.savefig('plots/confusion_matrix.png',bbox_inches='tight',dpi=300)
    plt.show()   