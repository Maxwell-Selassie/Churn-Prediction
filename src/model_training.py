def run(): 
    # import libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
    from sklearn.preprocessing import RobustScaler,PolynomialFeatures,OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.utils import compute_class_weight
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from xgboost import XGBClassifier
    import logging
    import warnings
    warnings.filterwarnings('ignore')

    logging.basicConfig(level=logging.INFO, format= '%(asctime)s - %(levelname)s : %(message)s', datefmt='%H:%M:%S')

    try:
        x_train = pd.read_parquet('../data/x_train.parquet')
        y_train = pd.read_parquet('../data/y_train.parquet')['y_train']
        logging.info('x_train and y_train files successfully opened!')
    except FileNotFoundError:
        logging.info('File Not Found! Please check filepath and try again!')
        raise


    # divide columns into numerical and categorical columns
    numeric_cols = x_train.select_dtypes(include='number').columns 
    categorical_cols = x_train.select_dtypes(include='object').columns

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42) # cross - validation

    # preprocessing
    num_pipe = Pipeline(steps=[
        ('impute',SimpleImputer(strategy='mean')),
        ('scaler',RobustScaler()),
        ('poly',PolynomialFeatures(include_bias=False))
    ])

    cat_pipe = Pipeline(steps=[
        ('impute',SimpleImputer(strategy='most_frequent')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num',num_pipe,numeric_cols),
        ('cat',cat_pipe,categorical_cols)
    ],verbose_feature_names_out=False)


    # sample weights to handle class imbalance
    sample_weights = compute_class_weight(class_weight='balanced',y=y_train,classes=np.unique(y_train))


    # models and hyperparameter tuning
    models_grid = {
        'Logistic_Regression' : {
        'model': LogisticRegression(penalty='l2',solver='lbfgs',random_state=42,n_jobs=-1,class_weight='balanced'),
        'params' : {
            'preprocessor__num__poly__degree' : [1,2],
            'classifier__C' : [0.5,0.8,1.0,3.0],
            'classifier__max_iter' : [400,500,600]
        }},
        'Random_Forest' : {
            'model' : RandomForestClassifier(n_jobs=-1,random_state=42,class_weight='balanced'),
            'params' : {
                'preprocessor__num__poly__degree' : [1,2],
                'classifier__n_estimators' : [100,120,150],
                'classifier__max_depth' : [6,8,10],
                'classifier__min_samples_split' : [3,5,7]
            }
        },
        'Decision_Trees' : {
            'model' : DecisionTreeClassifier(random_state=42,class_weight='balanced'),
            'params' : {
                'preprocessor__num__poly__degree' : [1,2],
                'classifier__max_depth' : [7,9,11],
                'classifier__min_samples_split' : [2,5,8]
            }
        },
        'XGBClassifier' : {
            'model' : XGBClassifier(objective='binary:logistic',verbosity=0,scale_pos_weight=sample_weights[0]/sample_weights[1]),
            'params' : {
                'preprocessor__num__poly__degree' : [1,2],
                'classifier__n_estimators' : [100,150,200],
                'classifier__max_depth' : [7,9,11],
                'classifier__learning_rate' : [0.3,0.5,0.7],
                'classifier__reg_lambda' : [0.1,0.3,0.5],
            }
        }
    }
    best_score = -float('inf')
    best_name = None
    best_estimator = None

    results = {}
    for name,model in models_grid.items():
        pipe = Pipeline(steps=[
            ('preprocessor',preprocessor),
            ('classifier',model['model'])
        ])

    # grid search
        model = GridSearchCV(estimator=pipe,
                        param_grid=model['params'],
                        cv = cv,
                        refit = True,
                        scoring= 'f1',
                        return_train_score=True,
                        n_jobs=-1,
                        verbose=2,
                        error_score='raise')
        
        # Training models
        print(f'Training model using {name}: (This may take a while)...')
        model.fit(x_train,y_train)

        results[name] = {
            'best_score' : model.best_score_,
            'best_params' : model.best_params_
        }
        if model.best_score_ > best_score:
            best_score = model.best_score_
            best_name = name
            best_estimator = model.best_estimator_

        # save models to joblib
        import joblib
        joblib.dump(model.best_estimator_,f'../models/{name}_best_model.pkl')
        print('Best model estimator successfully saved')

        # save results to a json file
        import json
        with open('../models/model_results.json','w') as file:
            json.dump(results,file,indent=4)

        print('-'*50)