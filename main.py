import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from utils.data import Data
from extraction.shape import Shape
from extraction.texture import Texture
from models.models import Catboost
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

import torch.nn.functional as F

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
#         self.fc1 = nn.Linear(64 * 12 * 12, 64)  # Adjust this based on the actual size after conv/pool
#         self.fc2 = nn.Linear(64, 2)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = x.view(-1, 64 * 12 * 12)  # Flatten the tensor
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x


def main():
    BATCH_SIZE = 4
    EPOCH = 15

    # Load data
    train_dataset = Data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(train_dataset.images_aug, train_dataset.labels_aug, test_size=.2, shuffle=True)

    # Ekstraksi Bentuk
    shape = Shape()
    # Ekstraksi bentuk menggunakan Canny
    X_train_shape = shape.create_tabular_data(X_train)
    X_test_shape = shape.create_tabular_data(X_test)
    cols = {'Feature_0': 'Perimeter',
            'Feature_1': 'Area',
            'Feature_2': 'Aspect Ratio',
            'Feature_3': 'Bounding Rectangle Width',
            'Feature_4': 'Bounding Rectangle Height'
            }
    X_train_shape.columns = [cols[col] if col in cols else col for col in X_train_shape.columns]
    X_test_shape.columns = [cols[col] if col in cols else col for col in X_test_shape.columns]

    # Ekstraksi Tekstur
    extraction = Texture()
    # Ekstraksi tekstur menggunakan GLCM
    X_train_texture = extraction.extract_features(X_train)
    X_test_texture = extraction.extract_features(X_test)

    # Merge Data hasil ekstraksi
    X_train = pd.concat([X_train_shape, X_train_texture], axis=1)
    X_test = pd.concat([X_test_shape, X_test_texture], axis=1)

    # impute
    X_train = X_train.interpolate()
    X_test = X_test.interpolate()

    # encoding
    encoding_map = {'no': 0, 'yes': 1}
    y_train = np.vectorize(encoding_map.get)(y_train)
    y_test = np.vectorize(encoding_map.get)(y_test)

    # normalisasi
    mms = MinMaxScaler()
    ss = StandardScaler()
    nor = Normalizer()
    rs = RobustScaler()

    X_train_scaled = mms.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)

    X_test_scaled = mms.fit_transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Modeling : Catboost
    catboost = Catboost()
    y_train_pred, y_val_pred = catboost.models(X_train_scaled, y_train, X_test_scaled, y_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_test, y_val_pred)

    print(f"Train Accuracy: {train_acc:.2f}")
    print(f"Validation Accuracy: {val_acc:.2f}")

    # # Best Parameters
    # best_params = catboost.best_param_meters(X_train_scaled, y_train)
    #
    # # Menggunakan Model Catboost dengan parameter terbaik
    # catboost = CatBoostClassifier(**best_params)
    # catboost.fit(X_train_scaled, y_train)
    # y_train_pred = catboost.predict(X_train_scaled)
    # y_val_pred = catboost.predict(X_test_scaled)
    #
    # train_acc = accuracy_score(y_train, y_train_pred)
    # val_acc = accuracy_score(y_test, y_val_pred)
    #
    # print(f"Train Accuracy: {train_acc:.2f}")
    # print(f"Validation Accuracy: {val_acc:.2f}")


if __name__=="__main__":
    main()