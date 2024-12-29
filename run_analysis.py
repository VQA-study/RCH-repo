import cv2
import numpy as np
from edge_detection import apply_chen_frei_edge_detection
from extract_contour import find_contours
from divide_shape import divide_and_label_contour


def process_image(image_path):
    image = cv2.imread(image_path)
    original = image.copy()

    edges = apply_chen_frei_edge_detection(image)
    contours = find_contours(edges)

    contour_image = np.zeros_like(original)
    for contour in contours:
        cv2.drawContours(contour_image, [contour], -1, (0, 0, 255), 1)
        divide_and_label_contour(contour_image, contour)

    cv2.imshow("Original Image", original)
    cv2.imshow("Chen-Frei Edges", edges)
    cv2.imshow('Outer edge and labeled shape sections', contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

process_image("sample_image_table.png")