import cv2
import numpy as np


def find_contours(edges):
    _, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def analyze_shape(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

    if circularity > 0.8:
        return "Round"
    elif 0.5 < circularity <= 0.8:
        return "Oval"
    else:
        return "Angular"