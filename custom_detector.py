import time
import numpy as np
from ultralytics.models.yolo.detect import DetectionPredictor


class CustomDetector(DetectionPredictor):
    def detect(self, frame):
        start = time.time()
        im = self.preprocess([frame])
        t_pre = (time.time() - start) * 1000

        start = time.time()
        preds = self.inference(im)
        t_inf = (time.time() - start) * 1000

        start = time.time()
        results = self.postprocess(preds, im, [frame])
        indices = non_max_suppression(results[0].boxes.cpu().numpy().xyxy, self.args.iou)
        t_post = (time.time() - start) * 1000


        results[0].speed['preprocess'] = t_pre
        results[0].speed['inference'] = t_inf
        results[0].speed['postprocess'] = t_post

        results[0] = results[0][indices]
        return results

def non_max_suppression(boxes, max_bbox_overlap, scores=None):

    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick