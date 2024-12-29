import cv2
import numpy as np
from extract_contour import analyze_shape


def divide_and_label_contour(image, contour):
    x, y, w, h = cv2.boundingRect(contour)  # 물체만 따로 분류 위한 바운딩 박스.
    third_height = h // 3  # 물체 '상부', '중부', '하부' 부분 나누기

    sections = {
        "Upper": (y, y + third_height),  
        "Middle": (y + third_height, y + 2 * third_height),
        "Lower": (y + 2 * third_height, y + h)
    }

    for section_name, (start, end) in sections.items():
        mask = np.zeros_like(image[:, :, 0])
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask[:start, :] = 0  # 상단, 중앙부, 하단부 구분 명확히 위한 y좌표 기준 이전 상단 부분 0으로 설정
        mask[end:, :] = 0  # y좌표 기준 이후 하단 부분 0으로 설정, 최종 현재 부분만 남기기

        section_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if section_contours:
            section_contour = max(section_contours, key=cv2.contourArea)
            section_shape = analyze_shape(section_contour)
            M = cv2.moments(section_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"]) + (-10 if section_name == "Upper" else (10 if section_name == "Lower" else 0))  # 출력 텍스트 위치 조정.
                cv2.putText(image, f"{section_name}: {section_shape}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)