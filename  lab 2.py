# lab 2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.spatial.distance import cosine

#  A1 

def load_purchase_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Purchase data")
    X = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = df["Payment (Rs)"].values.reshape(-1, 1)
    return X, y

def compute_rank(X):  #rank
    return np.linalg.matrix_rank(X)

def pseudoinverse(X, y): #pinv 
    X_pinv = np.linalg.pinv(X)
    return X_pinv @ y

#  A2 

def label_customers(y):  #labeliing
    return np.where(y.flatten() > 200, 1, 0)

def train_classifier(X, labels): #classifyinh theh data
    model = LogisticRegression()
    model.fit(X, labels)
    return model

#  A3 

def mean_numpy(data):  #mean
    return np.mean(data)

def var_numpy(data): # var
    return np.var(data)

def mean_manual(data):  #manually calc mean
        return sum(data) / len(data)

def var_manual(data):  #manually calc var
        mu = mean_manual(data)
        return sum((x - mu) ** 2 for x in data) / len(data)



#  A5 

def jaccard_coefficient(v1, v2):  # jaccard calculation
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    denom = f01 + f10 + f11
    if denom == 0:
        return 0.0
    return f11 / denom


def smc(v1, v2):  # smc calculation
    f11 = np.sum((v1 == 1) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    denom = f00 + f01 + f10 + f11
    if denom == 0:
        return 0.0
    return (f11 + f00) / denom

#  A6 

def cosine_similarity(v1, v2):  # cosine sim
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return 1 - cosine(v1, v2)

#  A7 

def similarity_matrices(data): #similar matrice
    data = data.astype(float)
    n = len(data)
    jc = np.zeros((n, n))
    smc_m = np.zeros((n, n))
    cos_m = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            jc[i, j] = jaccard_coefficient(data[i], data[j])
            smc_m[i, j] = smc(data[i], data[j])
            cos_m[i, j] = cosine_similarity(data[i], data[j])

    return jc, smc_m, cos_m

#  A8 

def impute_data(df):   #imputing data
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            if df[col].skew() < 1:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    return df

#  A9 

def normalize_data(df):  #normallizing he data from excel
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

#  MAIN 

def main():
    filepath = "/Users/agastya/Downloads/ML/ASS-2/Lab Session Data.xlsx"

    # ---------- A1 ----------
    X, y = load_purchase_data(filepath)
    rank = compute_rank(X)
    cost = pseudoinverse(X, y)

    print("Dimensionality:", X.shape[1])
    print("No of vectors:", X.shape[0])
    print("Rank :", rank)
    print("Prod costs:\n", cost.flatten())

    # ---------- A2 ----------
    labels = label_customers(y)
    model = train_classifier(X, labels)
    print("Classifier finished the traingng ")

    # ---------- A3 ----------
    stock = pd.read_excel(filepath, sheet_name="IRCTC Stock Price")
    prices = stock.iloc[:, 3].dropna().values

    print("Mean :", mean_numpy(prices))
    print("Variance :", var_numpy(prices))
    print("Mean :", mean_manual(prices))
    print("Variance :", var_manual(prices))


    wed_data = stock[stock["Day"] == "Wednesday"].iloc[:, 3]
    print("Wednesday Mean:", mean_numpy(wed_data))
    
    stock["Date"] = pd.to_datetime(stock["Date"])
    april_data = stock[stock["Date"].dt.month == 4].iloc[:, 3]
    print("April Mean:", mean_numpy(april_data))

    loss_prob = np.mean(stock["Chg%"].apply(lambda x: x < 0))
    print("Probability loss:", loss_prob)

    profit_wed = np.mean(stock[stock["Day"] == "Wednesday"]["Chg%"] > 0)
    print("Profit Wednesday:", profit_wed)

    plt.scatter(stock["Day"], stock["Chg%"])
    plt.title("Chg% vs Day")
    plt.show()

    # ---------- A4 ----------
    thyroid = pd.read_excel(filepath, sheet_name="thyroid0387_UCI")
    print(thyroid.info())
    print(thyroid.describe())

    # ---------- A5 ----------
    binary_data = thyroid.select_dtypes(include=["int64"]).head(20).values
    jc, smc_m, cos_m = similarity_matrices(binary_data)

    sns.heatmap(jc, annot=False)
    plt.title("Jaccard Coefficient")
    plt.show()
    # ---------- A6 ----------

    sns.heatmap(smc_m, annot=False)
    plt.title("SMC")
    plt.show()
    # ---------- A7 ----------

    sns.heatmap(cos_m, annot=False)
    plt.title("Cosine Similarity")
    plt.show()
    

    # ---------- A8 ----------
    thyroid = impute_data(thyroid)

    # ---------- A9 ----------
    thyroid = normalize_data(thyroid)
    print("Data imputation,normalization completed")

#  DRIVER 

if __name__ == "__main__":
    main()