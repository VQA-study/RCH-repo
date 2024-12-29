import cv2
import numpy as np

def apply_chen_frei_edge_detection(image):
    chen_frei_detector_gy = np.array([
        [-1, -np.sqrt(2), -1],
        [0, 0, 0],
        [1, np.sqrt(2), 1]
    ], dtype=np.float32)

    chen_frei_detector_gx = np.array([
        [-1, 0, 1],
        [-np.sqrt(2), 0, np.sqrt(2)],
        [-1, 0, 1]
    ], dtype=np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges_x = cv2.filter2D(gray, -1, chen_frei_detector_gy)
    edges_y = cv2.filter2D(gray, -1, chen_frei_detector_gx)
    edges = cv2.magnitude(edges_x, edges_y)
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return edges