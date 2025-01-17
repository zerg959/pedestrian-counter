import cv2
from ultralytics import YOLO
import numpy as np
from telegram import Bot
import time

def send_telegram_message(bot_token, chat_id, message):
    """Отправляет сообщение в Telegram."""
    try:
        bot = Bot(token=bot_token)
        bot.send_message(chat_id=chat_id, text=message)
        print("Сообщение в Telegram успешно отправлено!")
    except Exception as e:
        print(f"Ошибка отправки сообщения в Telegram: {e}")


def detect_pedestrian_traffic(video_path, bot_token, chat_id):
    """Распознает пешеходный трафик в видео и отправляет отчет в Telegram по завершении."""
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
        return

    tracked_ids = set()
    line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Проблемы с получением кадра")
            break

        results = model(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                if cls == 0 and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, b)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    obj_id = hash((center_x, center_y))

                    if obj_id not in tracked_ids and center_y > line_y:
                         tracked_ids.add(obj_id)
                         cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                         cv2.putText(frame, f"Person", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0), 2)

        cv2.line(frame, (0, line_y), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"People Count: {len(tracked_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Pedestrian Traffic", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    message = f"Подсчет завершен. Общее количество пешеходов: {len(tracked_ids)}"
    send_telegram_message(bot_token, chat_id, message)
    print(f"Общее количество людей: {len(tracked_ids)}")



if __name__ == "__main__":
    video_path = "http://YOUR_IP:8080/video"  # Замените на ваш URL IP камеры
    bot_token = "YOUR_TELEGRAM_BOT_TOKEN"  # Замените на токен вашего бота
    chat_id = "YOUR_TELEGRAM_CHAT_ID"    # Замените на ID вашего чата

    detect_pedestrian_traffic(video_path, bot_token, chat_id)
