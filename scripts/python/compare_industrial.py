import pandas as pd
import time
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor



# 1. Chargement des données
df = pd.read_csv("datasets/combined.csv")

# 2. Prétraitement 
df = df[df.iloc[:, -1] < 26.0] 

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

def run_benchmark(name, model):
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()
    
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    
    print(f"{name:15} | RMSE: {rmse:.6f} | Temps: {end_time - start_time:.4f}s")

print(f"{'Modèle':15} | {'RMSE':13} | {'Temps D\'exécution'}")
print("-" * 50)

# n_estimators=100, max_depth=4 (pour le Boosting), max_bin=256
run_benchmark("XGBoost Hist", XGBRegressor(
    tree_method='hist', 
    n_estimators=100, 
    max_depth=4, 
    max_bin=256,
    learning_rate=0.2,
    n_jobs=-1 
))

run_benchmark("LightGBM", LGBMRegressor(
    n_estimators=100, 
    max_depth=4, 
    max_bin=255, 
    learning_rate=0.2,
    n_jobs=-1
))


run_benchmark("Sklearn Hist", HistGradientBoostingRegressor(
    max_iter=100, 
    max_depth=4, 
    max_bins=255,
    learning_rate=0.2
))


# Le Random Forest est la version "HPC" du Bagging
run_benchmark("Sklearn RF", RandomForestRegressor(
    n_estimators=100, 
    max_depth=12, 
    n_jobs=-1 
))