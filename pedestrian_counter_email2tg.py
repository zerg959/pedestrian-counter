import cv2
from ultralytics import YOLO
from telegram import Bot
import imaplib
import email
import os
import re
import asyncio
from dotenv import load_dotenv
import hashlib

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

async def download_email_attachments(imap_server, imap_email, imap_password, download_dir, processed_hashes, video_extension='.mp4'):
    """Скачивает вложения из почты и возвращает путь к первому видеофайлу и тему письма."""
    try:
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(imap_email, imap_password)
        mail.select("inbox")

        _, data = mail.search(None, "ALL")
        for num in data[0].split():
            _, data = mail.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(data[0][1])
            subject = msg.get('Subject', 'Без темы')  # Получаем тему письма, если есть
            if "new_video" in subject.lower():
                for part in msg.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue
                    if part.get('Content-Disposition') is None:
                        continue

                    filename = part.get_filename()
                    if filename and filename.lower().endswith(video_extension):
                        sanitized_subject = sanitize_filename(subject)
                        base_name, ext = os.path.splitext(filename)
                        new_filename = f"{sanitized_subject}{ext}"
                        file_path = os.path.join(download_dir, new_filename)
                        with open(file_path, 'wb') as f:
                            f.write(part.get_payload(decode=True))
                        print(f"Файл {new_filename} скачан и сохранен в {download_dir}")

                        file_hash = calculate_file_hash(file_path)
                        if file_hash not in processed_hashes:
                           print(f"Файл с хешем {file_hash} ещё не обрабатывался. Продолжаем обработку.")
                           processed_hashes.add(file_hash)
                           mail.close()
                           mail.logout()
                           return file_path, new_filename, processed_hashes
                        else:
                            print(f"Файл с хешем {file_hash} уже был обработан. Пропускаем.")
                            try:
                                os.remove(file_path)
                                print(f"Файл {file_path} удален")
                            except Exception as e:
                                print(f"Ошибка удаления файла: {e}")
                            mail.close()
                            mail.logout()
                            return None, None, processed_hashes

            else:
                print(f"Пропущено письмо с темой: {subject}. Не найден ключ 'new_video'.")
                mail.close()
                mail.logout()
                return None, None, processed_hashes
        mail.close()
        mail.logout()
        return None, None, processed_hashes

    except Exception as e:
        print(f"Ошибка загрузки вложений: {e}")
        return None, None, processed_hashes


def detect_pedestrian_traffic(video_path):
    """Распознает пешеходный трафик в видео."""
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
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

    imap_server = os.getenv("IMAP_SERVER")
    imap_email = os.getenv("IMAP_EMAIL")
    imap_password = os.getenv("IMAP_PASSWORD")
    download_dir = os.getenv("DOWNLOAD_DIR")
    bot_token = os.getenv("BOT_TOKEN")
    chat_id = os.getenv("CHAT_ID")

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    processed_hashes = set()
    while True:
        video_path, video_filename, processed_hashes = await download_email_attachments(imap_server, imap_email, imap_password, download_dir, processed_hashes)

        if video_path:
            people_count, all_people_count = detect_pedestrian_traffic(video_path)
            if people_count is not None:
                message = f"Подсчет завершен.\nФайл: {video_filename}\nКоличество уникальных пешеходов (за весь отрезок): {people_count}, Общее количество обнаруженных пешеходов: {all_people_count}"
                print(f"Количество уникальных пешеходов (за весь отрезок): {people_count}, Общее количество обнаруженных пешеходов: {all_people_count}")
                await send_telegram_message(bot_token, chat_id, message)

            try:
                os.remove(video_path)
                print(f"Файл {video_path} удален")
            except Exception as e:
                print(f"Ошибка удаления файла: {e}")
        else:
            print("Новых видеофайлов с ключем не было получено. Ожидание...")

        await asyncio.sleep(30)

if __name__ == '__main__':
    asyncio.run(main())
