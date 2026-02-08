import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_regression import run_linear_regression
from decision_tree import run_decision_tree
from random_forest import run_random_forest

def main():
    dataset = pd.read_csv("data/housing.csv")
    dataset = dataset.drop(columns="ocean_proximity")
    dataset = dataset.dropna()

    #print(dataset.head())
    #print(dataset.isnull().sum())


    x = dataset.iloc[:, :-1]
    y = dataset["median_house_value"]

    cols = x.columns 

    sc = StandardScaler()
    x = sc.fit_transform(x)
    x = pd.DataFrame(x, columns=cols)


    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=42)


    lr_results = run_linear_regression(x_train, x_test, y_train, y_test)
    dt_results = run_decision_tree(x_train, x_test, y_train, y_test)
    rf_results = run_random_forest(x_train, x_test, y_train, y_test)

    results = [lr_results, dt_results, rf_results]

    for res in results:
        print(f"\nModel: {res['model_name']}")
        print(f"Train R² : {res['r2_train']:.3f}")
        print(f"Test R²  : {res['r2_test']:.3f}")
        print(f"RMSE     : {res['rmse']:.2f}")
        print(f"MAE      : {res['mae']:.2f}")

        plt.figure(figsize=(5,4))
        sns.scatterplot(x=res["y_true"], y=res["y_pred"])
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted ({res['model_name']})")
        plt.show()


if __name__ == '__main__':
    main()
    