import cv2
from ultralytics import YOLO
from telegram import Bot
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def send_telegram_message(bot_token, chat_id, message):
    """Отправляет сообщение в Telegram."""
    try:
        bot = Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message)
        print("Сообщение в Telegram успешно отправлено!")
    except Exception as e:
        print(f"Ошибка отправки сообщения в Telegram: {e}", exc_info=True)


def detect_pedestrian_traffic(video_source):
    """Распознает пешеходный трафик в видео."""
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_source}")
        return None

    all_tracked_ids = set()
    line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4)
    tracked_objects = {}
    frame_count = 0
    passed_people_count = 0
    frame_skip = 10 # увеличили интервал между кадрами в 10 раз

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            frame = cv2.resize(frame, (640, 480)) # размер можно менять
            results = model.track(frame, persist=True)
            boxes = results[0].boxes.data.tolist()
            tracks = results[0].boxes.id.tolist()

            if not boxes or not tracks:
                frame_count+=1
                continue
            for box, track in zip(boxes, tracks):
                x1, y1, x2, y2, id_ = list(map(int, box[:4])) + [int(track)]
                center_x = (x1 + x2) // 2
                cls = int(box[5])
                confidence = float(box[4])
                if cls == 0 and confidence > 0.7:
                    if id_ not in tracked_objects:
                        tracked_objects[id_] = {
                            "passed": False,
                            "initial_x": center_x
                        }
                    if not tracked_objects[id_]["passed"] and center_x > tracked_objects[id_]["initial_x"] and center_x > line_x:
                        all_tracked_ids.add(id_)
                        tracked_objects[id_]["passed"] = True
                        passed_people_count += 1
        frame_count += 1

    cap.release()
    return len(all_tracked_ids), passed_people_count

async def main():
    """Основная функция для получения почты, обработки видео и отправки отчета в Telegram."""
    bot_token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")
    video_source = os.getenv("VIDEO_SOURCE")  # Получаем URL потока из .env


    people_count, all_people_count = detect_pedestrian_traffic(video_source)
    if people_count is not None:
       message = f"Подсчет завершен.\nКоличество уникальных пешеходов (за весь отрезок): {people_count}, Общее количество обнаруженных пешеходов: {all_people_count}"
       print(f"Количество уникальных пешеходов (за весь отрезок): {people_count}, Общее количество обнаруженных пешеходов: {all_people_count}")
       await send_telegram_message(bot_token, chat_id, message)


if __name__ == '__main__':
    asyncio.run(main())
