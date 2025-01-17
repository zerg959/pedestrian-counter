import cv2
from ultralytics import YOLO
import numpy as np
from telegram import Bot
import imaplib
import email
import os
import time
import re
import asyncio
from dotenv import load_dotenv

load_dotenv()

def send_telegram_message(bot_token, chat_id, message):
    """Отправляет сообщение в Telegram."""
    try:
        bot = Bot(token=bot_token)
        bot.send_message(chat_id=chat_id, text=message)
        print("Сообщение в Telegram успешно отправлено!")
    except Exception as e:
        print(f"Ошибка отправки сообщения в Telegram: {e}")


def sanitize_filename(filename):
    """Очищает имя файла от недопустимых символов."""
    return re.sub(r'[^\w\.\-]', '_', filename)

async def download_email_attachments(imap_server, imap_email, imap_password, download_dir, video_extension='.mp4'):
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

            if "new_video" in subject.lower(): # Проверяем наличие ключа в нижнем регистре
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
                        mail.close()
                        mail.logout()
                        return file_path, new_filename
            else:
                print(f"Пропущено письмо с темой: {subject}. Не найден ключ 'new_video'.")
                mail.close()
                mail.logout()
                return None, None
        mail.close()
        mail.logout()
        return None, None

    except Exception as e:
        print(f"Ошибка загрузки вложений: {e}")
        return None, None


def detect_pedestrian_traffic(video_path):
    """Распознает пешеходный трафик в видео."""
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
        return None

    tracked_ids = set()
    line_y = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

    while True:
        ret, frame = cap.read()
        if not ret:
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
    return len(tracked_ids)

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

    while True:
      video_path, video_filename = await download_email_attachments(imap_server, imap_email, imap_password, download_dir)

      if video_path:
            people_count = detect_pedestrian_traffic(video_path)
            if people_count is not None:
                  message = f"Подсчет завершен.\n\
                           Файл: {video_filename}\n\
                           Количество пешеходов: {people_count}"
                  send_telegram_message(bot_token, chat_id, message)

            try:
                  os.remove(video_path)
                  print(f"Файл {video_path} удален")
            except Exception as e:
                print(f"Ошибка удаления файла: {e}")
      else:
          print("Новых видеофайлов с ключем не было получено. Ожидание...")

      await asyncio.sleep(300) # Пауза 5 минут (300 секунд)

if __name__ == '__main__':
    asyncio.run(main())
