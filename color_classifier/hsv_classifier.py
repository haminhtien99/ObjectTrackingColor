import numpy as np
import cv2
class HSVClassifier:
    def __init__(self):
        self.color_ranges = {
            'red': [
                [np.array([0, 120, 70], dtype=np.uint8), np.array([10, 255, 255], dtype=np.uint8)],
                [np.array([170, 120, 70], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8)],
                [np.array([0, 80, 50], dtype=np.uint8), np.array([10, 200, 150], dtype=np.uint8)],
                [np.array([170, 80, 50], dtype=np.uint8), np.array([180, 200, 150], dtype=np.uint8)],
            ],
            'yellow': [
                [np.array([20, 100, 100], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8)],
                [np.array([25, 40, 120], dtype=np.uint8), np.array([35, 100, 200], dtype=np.uint8)]
            ],
            'blue': [
                [np.array([100, 150, 50], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8)],
                [np.array([105, 80, 80], dtype=np.uint8), np.array([120, 150, 180], dtype=np.uint8)],
                [np.array([110, 80, 30], dtype=np.uint8), np.array([130, 150, 60], dtype=np.uint8)],
                [np.array([105, 70, 20], dtype=np.uint8), np.array([135, 120, 70], dtype=np.uint8)]
            ],
            'black': [
                [np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 50], dtype=np.uint8)],
                [np.array([0, 0, 10], dtype=np.uint8), np.array([180, 120, 40], dtype=np.uint8)],
                [np.array([0, 0, 20], dtype=np.uint8), np.array([180, 100, 60], dtype=np.uint8)]
            ],
            'white': [
                [np.array([0, 0, 200], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8)],
                [np.array([0, 0, 190], dtype=np.uint8), np.array([180, 40, 240], dtype=np.uint8)],
                [np.array([0, 0, 170], dtype=np.uint8), np.array([180, 50, 174], dtype=np.uint8)]
            ],
            'grey': [
                [np.array([0, 0, 50], dtype=np.uint8), np.array([180, 50, 200], dtype=np.uint8)]
            ]
        }

    def detect_single_color(self, image, color_name):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = self.color_ranges[color_name.lower()]

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        # Remove small noise and fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def detect_all_colors(self, image):
        """Detect all 6 colors in the image"""
        results = {}
        for color_name in self.color_ranges.keys():
            results[color_name] = self.detect_single_color(image, color_name)
        return results

    def get_color_hsv(self, bgr_color):
        """Convert BGR color to HSV for analysis"""
        color_array = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(color_array, cv2.COLOR_BGR2HSV)
        return hsv[0][0]

    def get_largest_contour_per_mask(self, mask):
        """Return largest contour and its area from a binary mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None, 0
        largest = max(contours, key=cv2.contourArea)
        return largest, cv2.contourArea(largest)

    def check_contour_touches_edges(self, contour, image_shape, min_points=5):
        """
        Check if two adjacent edges of image do not contant too many contour points
        """
        if contour is None:
            return True

        h, w = image_shape[:2]

        # Get all points from the contour
        points = contour.reshape(-1, 2)
        # Check each edge
        bottom_edge_count = np.sum(points[:, 1] == h - 1)  # y == height-1
        top_edge_count = np.sum(points[:, 1] == 0)         # y == 0
        left_edge_count = np.sum(points[:, 0] == 0)        # x == 0
        right_edge_count = np.sum(points[:, 0] == w - 1)   # x == width-1

        # Count how many edges have at least min_points
        edge_counts = [bottom_edge_count, left_edge_count, top_edge_count, right_edge_count]
        contant_points = []
        for count in edge_counts:
            if count > min_points:
                contant_points.append(1)
            else: contant_points.append(0)
        res = [contant_points[i]+contant_points[i-1] for i in range(len(contant_points))]
        return 2 in res


    def get_largest_color_contours(self, image, min_area=1000):
        """
        Returns list of (color_name, contour, area) contours, sorted by area desc.
        """
        found = []  # (color_name, contour, area, mask)

        for color_name in self.color_ranges.keys():
            mask = self.detect_single_color(image, color_name)
            contour, area = self.get_largest_contour_per_mask(mask)
            if contour is not None and area >= min_area:
                found.append((color_name, contour, area))

        if not found:
            return []
        found.sort(key=lambda x: x[2], reverse=True)

        return found


    def determine_ground_and_object(self, contours_info, image_shape):
        """
        Determine which contour is ground and which is object based on edge touching.

        Args:
            contours_info: List of (color_name, contour, area) for exactly two contours
            image_shape: Shape of the image (h, w)

        Returns:
            object_info: (color_name, contour, area) for object contour
        """
        # Determine ground and object based on edge touching
        # use get information max 3 contours
        for i, contour_info in enumerate(contours_info):
            contour_touches = self.check_contour_touches_edges(contour_info[1], image_shape)
            if not contour_touches:
                return contour_info
            if i == 2: break
        return contours_info[1] # else return second contour


    def classify_color(self, image):
        image = cv2.resize(image, (128, 128))
        contours = self.get_largest_color_contours(image)
        if len(contours) == 1:
            return contours[0][0]
        if len(contours) == 0:
            return None
        object = self.determine_ground_and_object(contours, image.shape)
        return object[0]

    def __call__(self, boxes, frame):
        colors = []
        track_ids = []
        for box in boxes:
            track_id = box.id
            if track_id is None:
                continue
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy.astype(np.int32)
            crop = frame[y1:y2, x1:x2, :]
            color = self.classify_color(crop)
            colors.append(color)
            track_ids.append(int(track_id.item()))
        if colors:
            return dict(zip(track_ids, colors))
        else:
            return {}
