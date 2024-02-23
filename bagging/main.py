import pandas as pd

from sklearn.datasets import load_iris

from classifier import Classifier


if __name__ == '__main__':
    iris_data = load_iris()
    data = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    data["species"] = [iris_data.target_names[x]for x in iris_data.target]

    X = data.drop("species", axis=1)
    y = data["species"]

    model = Classifier(max_samples=50)
    model.train(X, y)
    pred = model.predict(X)
    print(pred)
