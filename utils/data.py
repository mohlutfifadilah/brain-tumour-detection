import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import imgaug.augmenters as iaa

Size = 150
name_fol = ['no', 'yes']

class Data(Dataset):
    def __init__(self, path='/Users/mac/Documents/Pattern Recognition/brain-tumour-detection/dataset'):
        self.path = path
        self.images = []
        self.labels = []
        self.images_aug = []
        self.labels_aug = []

        self._load_data()

    def _load_data(self):
        for folder in name_fol:
            folder_path = os.path.join(self.path, folder)
            for file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, file)
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (Size, Size))
                self.images.append(image)
                self.labels.append(folder)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)

        # data augmentation
        aug = iaa.Sequential([
            iaa.Fliplr(),  # Flip
            iaa.Affine(rotate=(-30, 30)),  # Rotasi -30 - 30 degree
            iaa.GaussianBlur(sigma=(0, 1.0)),  # Blur antara 0 - 1
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)),
            # Noise prob 50%
        ])

        num_augmented_images = 4

        augmented_images = []
        augmented_labels = []

        for i in range(len(self.images)):
            current_image = self.images[i]
            current_label = self.labels[i]

            augmented_images.append(current_image)
            augmented_labels.append(current_label)

            for _ in range(num_augmented_images):
                augmented = aug(images=[current_image])[0]
                augmented_images.append(augmented)
                augmented_labels.append(current_label)

        self.images_aug = np.array(augmented_images)
        self.labels_aug = np.array(augmented_labels)

        self.images_aug = np.array([cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) for color_image in self.images_aug])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        feature = self.images[item]
        label = self.labels[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.int64)

if __name__=="__main__":
    data = Data()