import pickle
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


class Classifier:
    def __init__(self,
                 tree_max_depth=5,
                 n_estimators=50,
                 max_samples=1,
                 bootstrap=True,
                 oob_score=True,
                 random_state=100):
        self.model_path = "./pickle"
        self.save_model = "bagging_clf"
        self.bagging_classifier_model = BaggingClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=tree_max_depth),
            n_estimators=n_estimators,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state
        )

    def train(self, train_x, train_y):
        clf_model = self.bagging_classifier_model.fit(train_x, train_y)
        with open(f'{self.model_path}/{self.save_model}.pickle', 'wb') as wf:
            pickle.dump(clf_model, wf, pickle.HIGHEST_PROTOCOL)
        print(f"Out Of Bag Score : {clf_model.oob_score_}")
        print(f"Accuracy : {clf_model.score(train_x, train_y)}")
        print("finish model train")

    def predict(self, item_x):
        train_model_path = f'{self.model_path}/{self.save_model}.pickle'
        assert os.path.isfile(train_model_path), f"not found train model: {train_model_path}"

        with open(train_model_path, 'rb') as f:
            clf_trained_model = pickle.load(f)
        return clf_trained_model.predict(item_x)
