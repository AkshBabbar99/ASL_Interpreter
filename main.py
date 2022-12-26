import os.path

import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import time
from modules.asl_neural_network import AslNeuralNet
from modules.relative_paths import Path


labels = joblib.load(Path().LABELS)
asl_model = AslNeuralNet()
asl_model.load_state_dict(torch.load(Path().MODEL))
print(f'Model: {asl_model}')
print('Model Loaded')


def hand_capture(frame):
    # hand = frame[50:498, 50:498]
    hand = frame[100:324, 100:324]
    hand = cv2.resize(hand, (224, 224))
    return hand


videoCapture = cv2.VideoCapture(0)
if videoCapture.isOpened() is False:
    print('ERROR: Camera not opened or not found. Try again!')

width = int(videoCapture.get(3))
height = int(videoCapture.get(4))
vid_output = cv2.VideoWriter(os.path.join(Path().RESOURCES, 'output_vid.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))


while videoCapture.isOpened():
    ret, frame = videoCapture.read()

    # cv2.rectangle(frame, (50, 50), (498, 498), (20, 34, 255), 2)
    cv2.rectangle(frame, (100, 100), (324, 324), (20, 34, 255), 2)
    detect_hand = hand_capture(frame)

    target = detect_hand
    target = np.transpose(target, (2, 0, 1)).astype(np.float32)
    target = torch.tensor(target, dtype=torch.float)
    target = target.unsqueeze(0)

    outputs = asl_model(target)
    _, preds = torch.max(outputs.data, 1)

    # cv2.putText(frame, labels.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 225), 2)
    # cv2.putText(frame, labels.classes_[preds], (200, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 225), 2)
    cv2.putText(frame, labels.classes_[preds], (200, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('image', frame)
    vid_output.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoCapture.release()
cv2.destroyAllWindows()





