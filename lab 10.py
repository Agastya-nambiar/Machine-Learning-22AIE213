# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import shap
from lime.lime_tabular import LimeTabularExplainer


# LOAD DATA
def load_data(path):
    # Read  CSV 
    df = pd.read_csv(path)
    print("Columns in dataset:", df.columns.tolist())  # check what we have

    if 'Ward Name' in df.columns:
        df = df.drop(columns=['Ward Name'])

    if 'Ward NO' in df.columns:
        df = df.drop(columns=['Ward NO'])

    df = df.select_dtypes(include=[np.number])

    # Target variable
    target_column = 'Consumption in ML'

    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


# A1: CORRELATION HEATMAP
def plot_correlation(X):
    #  to see if any features are highly correlated
    plt.figure(figsize=(8,6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Feature Correlation Heatmap")
    plt.show()


# PCA FUNCTION (A2, A3)
def apply_pca(X, variance):
    # Standardise first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # variance
    pca = PCA(n_components=variance)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca


# MODEL TRAINING
def train_model(X, y):
    #  20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest 
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, r2, rmse, X_train, X_test


# A4: SEQUENTIAL FEATURE SELECTION
def sequential_fs(X, y, k_features):
    #  forward selection to pick the best k features
    model = RandomForestRegressor(random_state=42)

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=k_features,
        direction='forward'
    )

    X_new = sfs.fit_transform(X, y)
    return X_new


# A5: SHAP
def shap_explain(model, X_train):
    # TreeExplainer works with RandomForest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # Summary plot
    shap.summary_plot(shap_values, X_train)


# A5: LIME
def lime_explain(model, X_train, X_test):
    # LIME needs a training set to learn
    explainer = LimeTabularExplainer(
        training_data=X_train,
        mode='regression'
    )

    # first test instance 
    exp = explainer.explain_instance(
        X_test[0],
        model.predict
    )

    print("LIME for first test sample:")
    print(exp.as_list())


# MAIN
if __name__ == "__main__":

    file_path = "Household_Profile_Bengaluru_ason_01-03-2011.csv"

    X, y = load_data(file_path)

    # A1 – correlation heatmap
    plot_correlation(X)

    # A2: PCA  99% variance 
    X_pca_99 = apply_pca(X, 0.99)
    model_99, r2_99, rmse_99, Xtr, Xte = train_model(X_pca_99, y)

    # A3: PCA 95% variance 
    X_pca_95 = apply_pca(X, 0.95)
    model_95, r2_95, rmse_95, _, _ = train_model(X_pca_95, y)

    # A4: Sequential Feature Selection – pick top 2 features 
    k = min(2, X.shape[1] - 1) 
    X_sfs = sequential_fs(X, y, k_features=k)
    model_sfs, r2_sfs, rmse_sfs, _, _ = train_model(X_sfs, y)

    # A5: SHAP and LIME for one instance
    shap_explain(model_99, Xtr)
    lime_explain(model_99, Xtr, Xte)

    # RESULTS 
    print(" RESULTS")
    print(f"PCA 99%  - R2: {r2_99:.4f}, RMSE: {rmse_99:.4f}")
    print(f"PCA 95%  - R2: {r2_95:.4f}, RMSE: {rmse_95:.4f}")
    print(f"SFS      - R2: {r2_sfs:.4f}, RMSE: {rmse_sfs:.4f}")