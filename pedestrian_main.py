# use videoflow to count pedestrians

import cv2
from ultralytics import YOLO
import numpy as np

def detect_pedestrian_traffic(video_path):
    """
    Распознает пешеходный трафик в видео с IP-камеры.

    Args:
      video_path: URL потока IP-камеры.
    """

    # Загрузка модели YOLO
    model = YOLO("yolov8n.pt")

    # Захват видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видеопоток с IP-камеры {video_path}")
        return

    # Список для отслеживания идентификаторов
    tracked_ids = set()
    
    # Координаты линии
    line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2) # Примерно посередине по вертикали

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Проблемы с получением кадра с IP-камеры, проверяйте соединение")
            break  # Выход если видео закончилось

        # Обнаружение объектов
        results = model(frame)

        # Обработка результатов
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                cls = int(box.cls[0])  
                confidence = float(box.conf[0])

                if cls == 0 and confidence > 0.5:  # 0 - класс "человек"
                    x1, y1, x2, y2 = map(int, b)
                    
                    # Вычисление центра прямоугольника
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Идентификатор для каждого объекта (мы будем использовать приблизительный центр)
                    obj_id = hash((center_x, center_y))

                    # Проверяем, был ли этот объект уже посчитан
                    if obj_id not in tracked_ids and center_y > line_y: # считаем только тех, кто пересек линию в одном направлении
                        tracked_ids.add(obj_id)
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                        cv2.putText(frame, f"Person", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)

        cv2.line(frame, (0, line_y), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), line_y), (0, 0, 255), 2) # отображаем линию
        cv2.putText(frame, f"People Count: {len(tracked_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Pedestrian Traffic", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Выход по нажатию 'q'
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Общее количество людей: {len(tracked_ids)}")


if __name__ == "__main__":
    # Пример IP-адреса. ЗАМЕНИТЕ ЭТО НА СВОЙ АДРЕС!!!
    video_path = "http://192.168.1.100:8080/video"  # Пример. Замените!
    detect_pedestrian_traffic(video_path)
