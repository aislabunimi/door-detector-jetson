import cv2
import numpy as np
import roslibpy as roslibpy
import base64
import torchvision.transforms as T

import torch
from matplotlib import pyplot as plt

from door_detector.model import Model
from door_detector.utilities import transform, PostProcess
labels = {0: 'Closed door', 1: 'Open door'}
COLORS = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)


model = Model()

ros = roslibpy.Ros(host='192.168.192.62', port=9090)
ros.run()

def get_image(message):
    width, height = message['width'], message['height']
    base64_bytes = base64.b64decode(message['data'])
    frame = np.frombuffer(base64_bytes, dtype=np.uint8).reshape((height, width, -1))

    frame_for_model = transform(frame).unsqueeze(0)

    outputs = model(frame_for_model)

    post_processor = PostProcess()
    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([[height, width]]))

    [image_data] = processed_data

    keep = image_data['scores'] > 0.5

    for label, score, (xmin, ymin, xmax, ymax) in zip(image_data['labels'][keep], image_data['scores'][keep], image_data['boxes'][keep]):
        label = label.item()
        frame = cv2.rectangle(frame, (int(0.1 * width), int(0.1 * height)), (int(0.1 * width), int(0.1 * height)), COLORS[label], 2)

    cv2.imshow('door_classified', frame)


listener = roslibpy.Topic(ros, '/camera_down/rgb/image_raw', 'sensor_msgs/Image')
listener.subscribe(get_image)

publisher = roslibpy.Topic(ros, '/door_detector/image', 'sensor_msgs/Image')
publisher.advertise()

try:
    while True:
        pass
except KeyboardInterrupt:
    ros.terminate()


