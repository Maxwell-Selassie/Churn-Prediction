import shap
from sklearn.inspection import permutation_importance
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')
import numpy as np

def feature_importance(best_model):
    '''Displays the top 20 important features'''
    print(f'Explaining Model : {best_model}')

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

    plt.figure(figsize=(10,6))
    sns.barplot(x='importances', y='feature names', data= feature_imp.head(20))
    plt.title(f'{best_model} - Top 20 importance features', fontsize = 12, fontweight = 'bold')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png',bbox_inches='tight', dpi=300)
    plt.show()

    feature_imp.to_csv('eda_reports/feature_importance.csv', index=False)

def permutation_importance(best_model, x_test, y_test):
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    perm_results = permutation_importance(
        best_model, x_test, y_test, n_repeats = 10, random_state = 42, n_jobs = -1
    )
    perm_df = pd.DataFrame({
        'Features' : feature_names,
        'Importance' : perm_results.importances_mean
    }).sort_values(by='Importance',ascending=False)
    perm_df.to_csv('eda_reports/permutation_importance.csv',index=False)
    print("Permutation Importance saved!")

def shap(best_model,x_test):
    '''SHAP explanability'''
    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    explainer = shap.Explainer(best_model.named_steps['classifier'], best_model.named_steps['preprocessor'].transform(x_test))
    shap_values = explainer(best_model.named_steps['preprocessor'].transform(x_test))

    shap.summary_plot(shap_values, features = best_model.named_steps['preprocessor'].transform(x_test),
                    feature_names = feature_names, show = False)
    plt.title(f'{best_model} - SHAP summary plot')
    plt.savefig('plots/shap_summary.png', bbox_inches='tight', dpi=300)
    plt.show()