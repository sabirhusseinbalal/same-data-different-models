from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def run_random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_true = y_test

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    r2_train = model.score(x_train, y_train)*100
    r2_test = model.score(x_test, y_test)*100

    return {
        "model_name": "Random Forest",
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "y_true": y_true,
        "y_pred": y_pred
    }
