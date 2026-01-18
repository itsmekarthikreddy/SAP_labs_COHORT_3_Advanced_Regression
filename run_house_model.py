#!/usr/bin/env python3
"""
Reproducible script to preprocess data, run Ridge and Lasso with cross-validation,
compare models, list important predictors, evaluate doubling alpha effect,
and retrain Lasso excluding top-5 features. Outputs results to console and files.
This script is designed for the workspace where `train.csv` is present.
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess(df):
    df = df.copy()
    # Drop Id
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])

    # Separate target
    y = df['SalePrice']
    X = df.drop(columns=['SalePrice'])

    # Identify numeric and categorical
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Simple imputers and encoders
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Use sparse_output for compatibility with scikit-learn >=1.2
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    return X, y, preprocessor, numeric_cols, cat_cols


def fit_models(X, y, preprocessor, alphas=None, cv=5, random_state=42):
    # Use KFold
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # RidgeCV and LassoCV (alphas are inverse of lambda in sklearn terminology)
    if alphas is None:
        alphas = np.logspace(-3, 3, 50)

    ridge_cv = Pipeline(steps=[('pre', preprocessor), ('model', RidgeCV(alphas=alphas, cv=kf, scoring='neg_mean_squared_error'))])
    lasso_cv = Pipeline(steps=[('pre', preprocessor), ('model', LassoCV(alphas=alphas, cv=kf, random_state=random_state, max_iter=10000))])

    ridge_cv.fit(X, y)
    lasso_cv.fit(X, y)

    # Extract best alphas
    best_alpha_ridge = ridge_cv.named_steps['model'].alpha_
    best_alpha_lasso = lasso_cv.named_steps['model'].alpha_

    # Compute CV RMSE
    def cv_rmse(pipe):
        scores = cross_val_score(pipe, X, y, cv=kf, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores).mean()
        return rmse

    ridge_rmse = cv_rmse(ridge_cv)
    lasso_rmse = cv_rmse(lasso_cv)

    return {
        'ridge_cv': ridge_cv,
        'lasso_cv': lasso_cv,
        'best_alpha_ridge': float(best_alpha_ridge),
        'best_alpha_lasso': float(best_alpha_lasso),
        'ridge_rmse': float(ridge_rmse),
        'lasso_rmse': float(lasso_rmse),
    }


def get_feature_importance(pipe, numeric_cols, cat_cols, top_n=10):
    # Extract preprocessor outputs to map coefficients to feature names
    pre = pipe.named_steps['pre']
    model = pipe.named_steps['model']

    # numeric features remain in order
    num_feats = numeric_cols

    # get categories from onehot encoder
    ohe = None
    for name, trans, cols in pre.transformers:
        if name == 'cat':
            ohe = trans.named_steps['onehot']
            cat_in = cols
            break

    cat_feats = []
    if ohe is not None:
        try:
            cat_names = ohe.get_feature_names_out(cat_in)
            cat_feats = cat_names.tolist()
        except Exception:
            # fallback safe names
            cat_feats = [f"{c}_val{i}" for c in cat_in for i in range(1)]

    feat_names = num_feats + cat_feats

    coefs = model.coef_
    if coefs.shape[0] == 0:
        return []

    coef_abs = np.abs(coefs)
    sorted_idx = np.argsort(coef_abs)[::-1]
    top_features = [(feat_names[i], float(coefs[i])) for i in sorted_idx[:top_n] if i < len(feat_names)]
    return top_features


def retrain_with_doubled_alpha(pipe, X, y, double_alpha, model_type='ridge'):
    # Build pipeline with doubled alpha
    pre = pipe.named_steps['pre']
    if model_type == 'ridge':
        model = Ridge(alpha=double_alpha)
    else:
        model = Lasso(alpha=double_alpha, max_iter=10000)

    new_pipe = Pipeline(steps=[('pre', pre), ('model', model)])
    new_pipe.fit(X, y)
    return new_pipe


def main():
    base = Path(__file__).resolve().parent
    df = load_data(base / 'train.csv')
    X, y, preprocessor, numeric_cols, cat_cols = preprocess(df)

    print('Preprocessing complete. Numeric cols:', len(numeric_cols), 'Categorical cols:', len(cat_cols))

    results = fit_models(X, y, preprocessor)

    print('\nOptimal alphas:')
    print(' - Ridge alpha:', results['best_alpha_ridge'])
    print(' - Lasso alpha:', results['best_alpha_lasso'])
    print('\nCross-validated RMSE:')
    print(' - Ridge RMSE:', results['ridge_rmse'])
    print(' - Lasso RMSE:', results['lasso_rmse'])

    # Feature importances
    ridge_top = get_feature_importance(results['ridge_cv'], numeric_cols, cat_cols, top_n=10)
    lasso_top = get_feature_importance(results['lasso_cv'], numeric_cols, cat_cols, top_n=10)

    print('\nTop features (Ridge):')
    for f, c in ridge_top:
        print(' ', f, c)

    print('\nTop features (Lasso):')
    for f, c in lasso_top:
        print(' ', f, c)

    # Effect of doubling alpha
    ridge_double = retrain_with_doubled_alpha(results['ridge_cv'], X, y, results['best_alpha_ridge'] * 2, model_type='ridge')
    lasso_double = retrain_with_doubled_alpha(results['lasso_cv'], X, y, results['best_alpha_lasso'] * 2, model_type='lasso')

    def model_rmse(pipe):
        Xt = pipe.named_steps['pre'].transform(X)
        preds = pipe.named_steps['model'].predict(Xt)
        return float(np.sqrt(mean_squared_error(y, preds)))

    print('\nRMSE on training data after doubling alpha:')
    print(' - Ridge doubled alpha RMSE:', model_rmse(ridge_double))
    print(' - Lasso doubled alpha RMSE:', model_rmse(lasso_double))

    # Top-5 Lasso features to remove (clean names)
    top5_lasso = [f.strip() for f, _ in lasso_top[:5]]
    top5_lasso = [f for f in top5_lasso if f]
    print('\nTop-5 Lasso features to remove:', top5_lasso)

    # Retrain Lasso without top-5 features
    X_reduced = X.copy()
    for feat in top5_lasso:
        if feat in X_reduced.columns:
            X_reduced = X_reduced.drop(columns=[feat])
        else:
            if '_' in feat:
                base_col = feat.split('_')[0]
                if base_col in X_reduced.columns:
                    X_reduced = X_reduced.drop(columns=[base_col])

    # Rebuild preprocessor for reduced feature set
    numeric_cols_r = X_reduced.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols_r = X_reduced.select_dtypes(include=['object']).columns.tolist()

    numeric_transformer_r = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer_r = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor_r = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_r, numeric_cols_r),
            ('cat', categorical_transformer_r, cat_cols_r)
        ]
    )

    lasso_r = Pipeline(steps=[('pre', preprocessor_r), ('model', LassoCV(alphas=np.logspace(-3,3,50), cv=5, random_state=42, max_iter=10000))])
    lasso_r.fit(X_reduced, y)

    lasso_r_top = get_feature_importance(lasso_r, numeric_cols_r, cat_cols_r, top_n=10)
    print('\nTop features after removing top-5 Lasso predictors:')
    for f, c in lasso_r_top:
        print(' ', f, c)

    # Save summary
    out = {
        'best_alpha_ridge': results['best_alpha_ridge'],
        'best_alpha_lasso': results['best_alpha_lasso'],
        'ridge_rmse_cv': results['ridge_rmse'],
        'lasso_rmse_cv': results['lasso_rmse'],
        'ridge_top': ridge_top,
        'lasso_top': lasso_top,
        'top5_lasso_removed': top5_lasso,
        'lasso_retrained_top': lasso_r_top,
    }

    with open(base / 'model_summary.json', 'w') as f:
        json.dump(out, f, indent=2)

    print('\nSaved summary to model_summary.json')


if __name__ == '__main__':
    main()
