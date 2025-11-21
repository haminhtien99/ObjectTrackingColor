import torch
import cv2
from torch import nn
from torchvision import transforms
from torchvision import models
import numpy as np


class CnnClassifier:
    def __init__(self, model_path: str, device, classes, img_size=(224, 224)):

        self.device = device
        self.img_size = img_size
        self.classes = classes
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 7)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model

    def __call__(self, boxes, frame):
        crops = []
        track_ids = []
        for box in boxes:
            track_id = box.id
            if track_id is None:
                continue
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy.astype(np.int32)
            crop = frame[y1:y2, x1:x2, :]
            crops.append(crop)
            track_ids.append(int(track_id.item()))
        if crops:
            crop_preproces = self._preprocess(crops)
            with torch.no_grad():
                output = self.model(crop_preproces)
            indices = torch.argmax(output, dim=1).tolist()
            colors = [self.classes[i] for i in indices]
            return dict(zip(track_ids, colors))
        else:
            return {}

    def _preprocess(self, im_crops):
        # convert list image to tensor
        def _resize(im):
            return cv2.resize(im, self.img_size)
        im_batch = torch.cat([
            self.transform(_resize(im)).unsqueeze(0) 
            for im in im_crops
        ], dim=0).to(torch.float32)

        return im_batch


