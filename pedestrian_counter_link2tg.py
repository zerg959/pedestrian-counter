import cv2
import requests
from ultralytics import YOLO
from telegram import Bot
import os
import re
import asyncio
from dotenv import load_dotenv
import hashlib
import signal
import tempfile

load_dotenv()

async def send_telegram_message(bot_token, chat_id, message):
    """Отправляет сообщение в Telegram."""
    try:
        bot = Bot(token=bot_token)
        await bot.send_message(chat_id=chat_id, text=message)
        print("Сообщение в Telegram успешно отправлено!")
    except Exception as e:
        print(f"Ошибка отправки сообщения в Telegram: {e}", exc_info=True)

def sanitize_filename(filename):
    """Очищает имя файла от недопустимых символов."""
    return re.sub(r'[^\w\.\-]', '_', filename)


def calculate_file_hash(file_path):
    """Вычисляет SHA256 хеш файла."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Читаем файл по частям, чтобы не загрузить весь файл в память
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_video_from_url(url):
    """Скачивает видео по ссылке и возвращает путь к локальному файлу."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Проверяем, что запрос успешен

        # Создаем временный файл для сохранения видео
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_file_path = temp_file.name
        return temp_file_path
    except requests.exceptions.RequestException as e:
        print(f"Ошибка загрузки видео по ссылке: {e}")
        return None
    
def detect_pedestrian_traffic_from_url(video_path):
    """Распознает пешеходный трафик в видео."""
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
        return None, None

    all_tracked_ids = set()
    line_x = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4)
    tracked_objects = {}
    frame_count = 0
    passed_people_count = 0
    frame_skip = 10

    try:
      while True:
          ret, frame = cap.read()
          if not ret:
              break
          if frame_count % frame_skip == 0:
              frame = cv2.resize(frame, (640, 480))
              results = model.track(frame, persist=True)
              boxes = results[0].boxes.data.tolist()
              tracks = results[0].boxes.id.tolist()

              if not boxes or not tracks:
                  frame_count += 1
                  continue
              for box, track in zip(boxes, tracks):
                  x1, _, x2, _, id_ = list(map(int, box[:4])) + [int(track)]
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
    except KeyboardInterrupt:
       print("Обработка видеопотока остановлена пользователем.")
    finally:
      cap.release()
    return len(all_tracked_ids), passed_people_count


async def main():
    """Основная функция для обработки видеопотока с IP-камеры и отправки отчета в Telegram."""
    video_url = os.getenv("VIDEO_URL")
    bot_token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")


    video_path = download_video_from_url(video_url)
    if video_path:
        people_count, all_people_count = detect_pedestrian_traffic_from_url(video_path)
        if people_count is not None:
            message = f"Подсчет завершен.\nКоличество уникальных пешеходов: {people_count}, Общее количество обнаруженных пешеходов: {all_people_count}"
            print(f"Количество уникальных пешеходов: {people_count}, Общее количество обнаруженных пешеходов: {all_people_count}")
            await send_telegram_message(bot_token, chat_id, message)
    else:
        print("Не удалось обработать видеопоток. Проверьте URL камеры и ее доступность.")



if __name__ == '__main__':
    asyncio.run(main())