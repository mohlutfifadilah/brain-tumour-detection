import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.util import img_as_ubyte
from skimage.measure import shannon_entropy
from torch.utils.data import Dataset

class Texture(Dataset):
    def __init__(self,):
        pass

    def extract_features(self, dataset):
        feature_dataset = pd.DataFrame()

        for image in range(dataset.shape[0]):
            df = pd.DataFrame()

            img = dataset[image, :, :]

            # Convert the image to unsigned integer type
            img = img_as_ubyte(img)

            # Ekstraksi fitur Sobel
            sobel_image = sobel(img)

            # Ekstraksi fitur GLCM pada citra asli
            GLCM = graycomatrix(img, [1], [0])
            GLCM_Energy = graycoprops(GLCM, 'energy')[0]
            df['Energy'] = GLCM_Energy
            GLCM_corr = graycoprops(GLCM, 'correlation')[0]
            df['Corr'] = GLCM_corr
            GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
            df['Diss_sim'] = GLCM_diss
            GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
            df['Homogen'] = GLCM_hom
            GLCM_contr = graycoprops(GLCM, 'contrast')[0]
            df['Contrast'] = GLCM_contr
            GLCM_entropy = shannon_entropy(GLCM)
            df['Entropy'] = GLCM_entropy

            # Ekstraksi fitur GLCM pada citra hasil Sobel
            sobel_image = sobel(img)
            sobel_image_uint8 = img_as_ubyte(sobel_image)  # Convert to unsigned byte
            GLCM_sobel = graycomatrix(sobel_image_uint8, [3], [0])

            GLCM_Energy_sobel = graycoprops(GLCM_sobel, 'energy')[0]
            df['Energy_sobel'] = GLCM_Energy_sobel
            GLCM_corr_sobel = graycoprops(GLCM_sobel, 'correlation')[0]
            df['Corr_sobel'] = GLCM_corr_sobel
            GLCM_diss_sobel = graycoprops(GLCM_sobel, 'dissimilarity')[0]
            df['Diss_sim_sobel'] = GLCM_diss_sobel
            GLCM_hom_sobel = graycoprops(GLCM_sobel, 'homogeneity')[0]
            df['Homogen_sobel'] = GLCM_hom_sobel
            GLCM_contr_sobel = graycoprops(GLCM_sobel, 'contrast')[0]
            df['Contrast_sobel'] = GLCM_contr_sobel
            GLCM_entropy_sobel = shannon_entropy(GLCM_sobel)
            df['Entropy_sobel'] = GLCM_entropy_sobel

            feature_dataset = pd.concat([feature_dataset, df], ignore_index=True)

        return feature_dataset

if __name__ == "__main__":
    data = Texture()
