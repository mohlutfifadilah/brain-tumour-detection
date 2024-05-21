import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Shape(Dataset):
    def __init__(self):
        pass

    def extract_shape_features(self, image):
        edges = cv2.Canny(image, 100, 500)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Inisialisasi list untuk menyimpan fitur
        features = []

        # Loop melalui setiap kontur
        for contour in contours:
            # Hitung panjang lengkungan (perimeter)
            perimeter = cv2.arcLength(contour, True)

            # Hitung luas kontur
            area = cv2.contourArea(contour)

            # Hitung bentuk persegi panjang yang melingkupi kontur
            x, y, w, h = cv2.boundingRect(contour)

            # Hitung aspek rasio kontur
            aspect_ratio = float(w) / h if h != 0 else 0

            # Simpan fitur ke dalam list
            features.extend([perimeter, area, aspect_ratio, w, h])

        return features

    def create_tabular_data(self, images):
        # Mengekstraksi fitur dari setiap gambar
        all_features = []
        for image in images:
            features = self.extract_shape_features(image)
            all_features.append(features)

        # Memastikan bahwa semua baris memiliki jumlah fitur yang sama
        max_features = max(len(features) for features in all_features)
        for features in all_features:
            while len(features) < max_features:
                features.append(np.nan)

        # Menyusun data tabular
        columns = [f'Feature_{i}' for i in range(max_features)]
        df = pd.DataFrame(all_features, columns=columns)

        # Memilih hanya 5 kolom yang diinginkan
        selected_columns = df.iloc[:, :5]

        return selected_columns

if __name__=="__main__":
    data = Shape()


