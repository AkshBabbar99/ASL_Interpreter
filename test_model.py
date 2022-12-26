import joblib
import cv2
import os

import numpy as np
import torch
import albumentations
from modules.relative_paths import Path
from modules.asl_neural_network import AslNeuralNet


augmentation = albumentations.Compose([albumentations.Resize(224, 224, always_apply=True), ])
labels = joblib.load(Path().LABELS)
asl_model = AslNeuralNet()
asl_model.load_state_dict(torch.load(Path().MODEL))
print(asl_model)
print("ASL Model load successful")

for image_path in sorted(os.listdir(Path().TEST_IMAGES)):
    print('Starting Tests')
    root = Path().TEST_IMAGES
    image = cv2.imread(os.path.join(root, image_path))
    copy_image = image.copy()

    print(f'\nTesting: {image_path}')
    image = augmentation(image=np.array(image))['image']
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)
    image = image.unsqueeze(0)

    output = asl_model(image)
    _, preds = torch.max(output.data, 1)
    print(f'Predicted Output: {labels.classes_[preds]}')

    cv2.putText(copy_image, labels.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 225), 2)
    cv2.imshow('image', copy_image)
    cv2.imwrite(os.path.join(Path().TEST_OUTPUT, f'out_{image_path}'), copy_image)
