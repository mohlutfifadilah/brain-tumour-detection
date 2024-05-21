import cv2
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from torch.utils.data import Dataset

class Catboost(Dataset):
    def __init__(self):
        pass

    def models(self, X_train, y_train, X_test, y_test):
        catboost_model = CatBoostClassifier()

        catboost_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50, verbose=100)

        y_train_pred = catboost_model.predict(X_train)

        y_val_pred = catboost_model.predict(X_test)

        return y_train_pred, y_val_pred

    def best_param_meters(self, X_train, y_train):
        self.model = CatBoostClassifier()
        self.params = {
            'depth': [2, 4],
            'learning_rate': [0.01, 0.5],
            'iterations': [100, 200]
        }
        grid_search = GridSearchCV(self.model, self.params, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_params = grid_search.best_params_
        return best_params

if __name__=="__main__":
    data = Catboost()