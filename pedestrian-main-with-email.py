import cv2
from ultralytics import YOLO
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(sender_email, sender_password, receiver_email, subject, message, smtp_server, smtp_port):
    """
    Отправляет электронное письмо.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain'))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print("Электронное письмо успешно отправлено!")

    except Exception as e:
        print(f"Ошибка отправки электронного письма: {e}")


def detect_pedestrian_traffic(video_path, sender_email, sender_password, receiver_email, smtp_server, smtp_port):
    """
    Распознает пешеходный трафик в видео и отправляет отчет по завершении.
    """
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
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.line(frame, (0, line_y), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), line_y), (0, 0, 255), 2)
        cv2.putText(frame, f"People Count: {len(tracked_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Pedestrian Traffic", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    message = f"Подсчет завершен. Общее количество пешеходов: {len(tracked_ids)}"
    send_email(sender_email, sender_password, receiver_email, "Отчет о пешеходном трафике", message, smtp_server, smtp_port)
    print(f"Общее количество людей: {len(tracked_ids)}")


if __name__ == "__main__":
    video_path = "http://YOUR_IP:8080/video"  # Замените на ваш URL IP камеры
    sender_email = "your_email@gmail.com"  # Ваш электронный адрес
    sender_password = "your_password"  # Пароль от вашего электронного адреса
    receiver_email = "receiver_email@example.com"  # Адрес получателя
    smtp_server = "smtp.gmail.com"  # SMTP-сервер Gmail
    smtp_port = 587  # порт SMTP Gmail

    detect_pedestrian_traffic(video_path, sender_email, sender_password, receiver_email, smtp_server, smtp_port)
