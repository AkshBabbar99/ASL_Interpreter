import os

import albumentations
import cv2
import random
import joblib
import pandas as pd
import numpy as np
import torch

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from tqdm import tqdm

import relative_paths


class ImagePreprocessor:
    def __init__(self):
        self.root_training_path = relative_paths.Path().TRAINING_IMAGES
        self.alphabet_dir = sorted(os.listdir(self.root_training_path))
        self.preprocess_images_path = relative_paths.Path().PREPROCESSED_IMAGES

    def preprocess_images(self, num_images=2000):
        print(f'Preprocessing {num_images} images for each alphabet...')
        for num, dir_name in tqdm(enumerate(self.alphabet_dir), total=len(self.alphabet_dir)):
            if dir_name == '.DS_Store':
                continue
            image_folder_path = os.path.join(self.root_training_path, dir_name)
            list_images = os.listdir(image_folder_path)
            os.makedirs(os.path.join(self.preprocess_images_path, dir_name), exist_ok=True)
            for i in range(num_images):
                total_images = len(list_images)
                random_image_id = random.randint(0, total_images - 1)
                read_image = cv2.imread(f'{image_folder_path}/{list_images[random_image_id]}')
                read_image = cv2.resize(read_image, (224, 224))
                cv2.imwrite(os.path.join(self.preprocess_images_path, f'{dir_name}/{dir_name}{i}.jpg'), read_image)
        print('Image Preprocessing Completed')


class Mapper:
    def __init__(self):
        self.image_dir_paths = list(paths.list_images(relative_paths.Path().PREPROCESSED_IMAGES))
        self.df = pd.DataFrame()
        self.labels = self.create_labels()
        self.df_path = relative_paths.Path().DATAFRAME
        self.label_path = relative_paths.Path().LABELS
        self.binarizer = LabelBinarizer()

    def create_labels(self):
        print('Creating Labels...')
        alphabets = list()
        for idx, image_path in tqdm(enumerate(self.image_dir_paths), total=len(self.image_dir_paths)):
            alphabet = image_path.split(os.path.sep)[-2]
            self.df.loc[idx, 'image_path'] = image_path
            alphabets.append(alphabet)
        return alphabets

    def one_hot_encoding(self):
        print('One Hot Encoding the data...')
        self.labels = np.array(self.labels)
        self.labels = self.binarizer.fit_transform(self.labels)
        for i in range(len(self.labels)):
            index = np.argmax(self.labels[i])
            self.df.loc[i, 'target'] = int(index)
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df.to_csv(self.df_path, index=False)
        joblib.dump(self.binarizer, self.label_path)
        print(self.df.head(5))
        print('Labeling Completed')


ImagePreprocessor().preprocess_images(2000)
Mapper().one_hot_encoding()