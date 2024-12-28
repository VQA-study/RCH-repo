import cv2
import numpy as np



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


image = cv2.imread("sample_image_table.png")
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = gray.astype(np.float32)

edges_x = cv2.filter2D(gray, -1, chen_frei_detector_gy)
edges_y = cv2.filter2D(gray, -1, chen_frei_detector_gx)
edges = cv2.magnitude(edges_x, edges_y)

edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

_, binary_edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 외곽선 탐지, CHAIN_APPROX_SIMPLE: 노이즈 점 제거.

contour_image = np.zeros_like(original)  # 검은 빈 화면 생성.
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)  # 녹색 선 설정.

cv2.imshow("Original Image", original)
cv2.imshow("Chen-Frei Edges", edges)
cv2.imshow('Outer edge', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()