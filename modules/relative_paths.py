import os


class Path:
    def __init__(self):
        self.ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
        self.DATABASE = os.path.join(self.ROOT, 'database')
        self.TRAINING_IMAGES = os.path.join(self.DATABASE, 'train/alphabets')
        self.TEST_IMAGES = os.path.join(self.DATABASE, 'test/alphabets')
        self.PREPROCESSED_IMAGES = os.path.join(self.DATABASE, 'preprocessed_images')
        self.RESOURCES = os.path.join(self.ROOT, 'resources')
        self.LABELS = os.path.join(self.RESOURCES, 'labels.pkl')
        self.DATAFRAME = os.path.join(self.RESOURCES, 'data.csv')
        self.MODEL = os.path.join(self.RESOURCES, 'model.pth')
        self.TEST_OUTPUT = os.path.join(self.RESOURCES, 'test_output')
