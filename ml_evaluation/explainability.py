import shap
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import seaborn as sns

def compute_feature_importance(best_model):
    '''Displays the top 20 important features'''

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
        importances = best_model.named_steps['classifier'].feature_importances_
    elif hasattr(best_model.named_steps['classifier'], 'coef_'):
        importances = np.abs(best_model.named_steps['classifier'].coef_).flatten()
    else:
        importances = np.zeros(len(feature_names))


    feature_imp = pd.DataFrame({
        'feature names' : feature_names,
        'importances' : importances
    }).sort_values(by='importances', ascending=False)
    plt.figure(figsize=(15,8))
    sns.barplot(data=feature_imp.head(10), x='feature names',y='importances',color='indigo')
    plt.title('Top 10 Feature Importance')
    plt.xticks(rotation=90)
    plt.savefig('plots/feature_importance.png',bbox_inches='tight',dpi=300)
    plt.show()

    feature_imp.to_csv('eda_reports/feature_importance.csv', index=False)

def compute_permutation_importance(best_model, x_test, y_test):
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    perm_results = permutation_importance(
        best_model, x_test, y_test
    )
    perm_df = pd.DataFrame({
        'Features' : feature_names,
        'Importance' : perm_results.importances_mean
    }).sort_values(by='Importance',ascending=False)
    perm_df.to_csv('eda_reports/permutation_importance.csv',index=False)
    print("Permutation Importance saved!")

def compute_shap(best_model,x_test):
    '''SHAP explanability'''
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    explainer = shap.Explainer(best_model.named_steps['classifier'], best_model.named_steps['preprocessor'].transform(x_test))
    shap_values = explainer(best_model.named_steps['preprocessor'].transform(x_test))

    shap.summary_plot(shap_values, features = best_model.named_steps['preprocessor'].transform(x_test),
                    feature_names = feature_names, show = False)
    plt.title(f'SHAP summary plot')
    plt.savefig('plots/shap_summary.png', bbox_inches='tight', dpi=300)
    plt.show()