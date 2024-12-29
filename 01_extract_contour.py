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
cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 1)  # 빨간 선 설정.

for contour in contours:
    perimeter = cv2.arcLength(contour, True)  # Contour 둘레 계산.
    area = cv2.contourArea(contour)  # Contour 면적 계산.

    if perimeter > 0:
        circularity = 4 * np.pi * (area / (perimeter ** 2))  # Circularity 계산. 1에 가까우면 원형 가까운 형태
    else:
        circularity = 0


    if circularity > 0.8:
        shape = "Round"  # 원형
    elif 0.5 < circularity <= 0.8:
        shape = "Oval"  # 타원형
    else:
        shape = "Angular"  # 각진


    M = cv2.moments(contour)  # Contour 중심 기준 모멘텀 계산.
    if M["m00"] != 0:  # m00: Contour 면적(Contour 내부 픽셀값 합.)
        cx = int(M["m10"] / M["m00"])  # m10: Contour 내부 x좌표값 합.
        cy = int(M["m01"] / M["m00"])  # m01: Contour 내부 y좌표값 합.
        # 최종 각 좌표의 중심점 계산.
        cv2.putText(contour_image, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

cv2.imshow("Original Image", original)
cv2.imshow("Chen-Frei Edges", edges)
cv2.imshow('Outer edge and shape structure', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()