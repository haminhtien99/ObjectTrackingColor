"""
Pipeline:
Video -> Detector -> Filter by mask --> Tracking
                                    --> Classification by Color
Цвета: белый, серый, синий, черный, красный, желтый
"""


import json
import os
import cv2
import numpy as np
import time
import torch
from tqdm import tqdm

from trackers.custom_track import TRACKER_MAP
from trackers.utils.load_yaml import load_yaml
from ultralytics.utils import ASSETS

from custom_detector import CustomDetector
from color_classifier import ColorClassifier


def load_mask(datapath: str):
    # Get points for mask markup
    maskpath = os.path.join(datapath, "mask_markup.json")
    with open(maskpath, 'r') as f:
        markup = json.load(f)
    polygon = markup["areas"][0]

    videopath = os.path.join(datapath, "1.mp4")
    cap = cv2.VideoCapture(videopath)
    ret, first_frame = cap.read()
    cap.release()
    if ret:
        h, w = first_frame.shape[:2]
        pts = np.array([(int(pt[0]*w), int(pt[1]*h)) for pt in polygon], dtype=np.int32)
        return pts
    else:
        return None

class VehicleProcessing:
    def __init__(self, polygon, cfg_file, predictor, output_file):
        self.polygon = polygon

        cfg = load_yaml(cfg_file)
        self.tracker = TRACKER_MAP[cfg.tracker_type](cfg, frame_rate=30)
        self.predictor: CustomDetector = predictor

        self.output = output_file
        self.frame_id = 0
        self.color_classifier = ColorClassifier()
        self.color_dict = {}

    def update(self, frame):
        self.frame_id += 1

        # Detection
        det_results = self.predictor.detect(frame)[0]

        # Filter object inside mask and tracking
        keeps = self.filter_obj(det_results)
        results = self.tracking(det_results, keeps, frame)

        boxes = results.boxes.cpu().numpy()
        self.write_to_output(boxes)

        # Classification
        classify_time = self.classify_color(frame, boxes)
        results.speed['classify_time'] = classify_time
        drawed_frame = self.draw(frame, boxes)

        return results, drawed_frame

    def filter_obj(self, results):
        keeps = []
        boxes = results.boxes.cpu().numpy()
        if len(boxes) == 0:
            return keeps

        xyxy = boxes.xyxy
        for i in range(len(boxes)):
            xc = xyxy[i][0]/2 + xyxy[i][2]/2
            yc = xyxy[i][1]/2 + xyxy[i][3]/2
            inside = cv2.pointPolygonTest(self.polygon, (xc, yc), False) >= 0
            if inside:
                keeps.append(i)
        return keeps

    def tracking(self, det_results, keeps, frame):
        results = det_results[keeps]
        boxes = results.boxes.cpu().numpy()

        start = time.time()
        tracks = self.tracker.update(boxes, frame)
        if len(tracks) == 0:
            results.speed['association'] = 0.0
            return results
        association_time = (time.time() - start) * 1000
        idx = tracks[:, -1].astype(int)
        valid_indices = idx[idx > -1]   # hide unmatched track-id
        results = results[valid_indices]

        update_args = {"boxes": torch.as_tensor(tracks[:, :-1])}
        results.update(**update_args)
        results.speed['association'] = association_time
        return results

    def write_to_output(self, boxes):
        lines = []

        for box in boxes:
            xyxy = box.xyxy[0]
            if box.id is None:
                continue
            track_id = box.id.item()
            conf = box.conf.item()

            x, y = xyxy[:2]
            w = xyxy[2] - x
            h = xyxy[3] - y

            line = f"{self.frame_id},{int(track_id)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.2f},-1,-1,-1\n"
            lines.append(line)
            if self.frame_id == 1:
                mode = "w"
            else: mode = "a"
        with open(self.output, mode) as f:
            f.writelines(lines)

    def classify_color(self, frame, boxes):
        start = time.time()
        clone = frame.copy()

        h, w = frame.shape[:2]
        for box in boxes:
            if box.id is None:
                continue
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy.astype(np.int32)

            # expand box get more information from background
            x1 = max(x1 - 10, 0)
            y1 = max(y1 - 10, 0)
            x2 = min(x2 + 10, w)
            y2 = min(y2 + 10, h)
            crop = clone[y1:y2, x1:x2, :]

            track_id = int(box.id.item())
            color = self.color_classifier.detect_image_color(crop)
            self.color_dict[track_id] = color

            # save cropped to visualization
            output_folder = self.output.split(".")[0]
            name_crop = f"{track_id:03d}_{color}.jpg"
            crop_folder = os.path.join(output_folder, 'objects')
            if not os.path.exists(crop_folder):
                os.makedirs(crop_folder)
            cv2.imwrite(os.path.join(crop_folder, name_crop), crop)

        classify_time = time.time() - start
        return classify_time*1000

    def draw(self, frame, boxes):
        names = self.predictor.model.names
        for box in boxes:
            if box.id is None:
                continue
            track_id = box.id.item()
            cls = box.cls[0]
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy.astype(np.int32)
            color = self.color_dict[track_id]
            # draw frame with color object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{int(track_id)}-{names[cls]}-{color}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.polylines(frame, [self.polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        return frame


if __name__ == "__main__":
    cfg = load_yaml('track_cfg.yaml', return_dict=True)
    datafolder = 'data'
    show = cfg.pop('show')
    track_cfg = cfg.pop('track_cfg')
    videos = os.listdir(datafolder)
    name = videos[cfg.pop('video_id')]
    datapath = os.path.join(datafolder, name)
    videopath = os.path.join(datafolder, name, '1.mp4')

    output_file = os.path.join('outputs', f"{name}.txt")

    predictor = CustomDetector(overrides=cfg)
    # warmup
    for img in os.listdir(ASSETS):
        predictor(os.path.join(ASSETS, img))

    polygon = load_mask(datapath)
    vehicle_process = VehicleProcessing(polygon, track_cfg, predictor, output_file)
    cap = cv2.VideoCapture(videopath)
    print(f"{'preprocess':>15}{'inference':>15}{'postprocess':>15}{'association':>15}{'classify_time':>15}")
    pbar = tqdm(total=None)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (960, 540))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res, processed_frame = vehicle_process.update(frame)

        speed = res.speed
        pbar.set_description(
            f"{speed['preprocess']:>13.2f}ms "
            f"{speed['inference']:>13.2f}ms "
            f"{speed['postprocess']:>13.2f}ms "
            f"{speed['association']:>13.2f}ms "
            f"{speed['classify_time']:>13.2f}ms"
        )
        resize = cv2.resize(processed_frame, (0, 0), fx=0.5, fy=0.5)
        out.write(resize)
        if show:
            cv2.imshow("Video tracking", resize)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

