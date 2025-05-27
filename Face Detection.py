# Smart Detection - Phat hien nguoi la bang YOLOv10 va nhan dien khuon mat
import cv2
import time
import threading
import queue
import requests
import os
import datetime
from simple_facerec import SimpleFacerec
from ultralytics import YOLO
import torch

# Webhook de gui canh bao len Discord
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1376485779043520582/BjIkQdNhMBClmsVkXsGkxTWWjeVtiw_hj0SFThlBC3R2qWxeyDrwok5NjXktBv5g-Hch"

# ID cua class "person" trong tap du lieu COCO
PERSON_CLASS_ID = 0

# Khoi tao nhan dien khuon mat
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load model YOLO (su dung CUDA neu co GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO("yolov10n.pt").to(device)

# Khoi tao hang doi va lock cho multithreading
frame_queue = queue.Queue(maxsize=1)
processed_frame = None
frame_lock = threading.Lock()

# Gui anh canh bao khi phat hien nguoi la len Discord
def send_discord_alert(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"content": "CANH BAO: PHAT HIEN NGUOI LA!"}
            requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
    except Exception as e:
        print("Loi gui Discord:", e)

# Giao dien chup va luu khuon mat moi
def capture_new_face_by_id(cam_id, old_cap):
    print(f"\nDang mo camera {cam_id} de chup khuon mat...")
    old_cap.release()
    new_cap = cv2.VideoCapture(cam_id)

    if not new_cap.isOpened():
        print("Khong mo duoc camera.")
        return cv2.VideoCapture(0)

    window_name = "Camera"
    print("Nhan SPACE de chup, ESC de huy...")

    window_created = False

    while True:
        ret, frame = new_cap.read()
        if not ret:
            continue

        cv2.putText(frame, "Nhan SPACE de chup, ESC de huy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow(window_name, frame)
        window_created = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Huy chup.")
            break
        elif key == 32:  # SPACE
            name = input("Nhap ten nguoi moi: ")
            path = f"images/{name}.jpg"
            resized_frame = cv2.resize(frame, (640, 480))
            cv2.imwrite(path, resized_frame)
            print(f"Da luu anh {path}")
            sfr.load_encoding_images("images/")
            break

    new_cap.release()
    if window_created:
        cv2.destroyAllWindows()

    return cv2.VideoCapture(0)

# Luong xu ly nhan dien
def detection_worker():
    global processed_frame
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Resize nho de tang toc YOLO
        small = cv2.resize(frame, (208, 160))
        results = yolo_model.predict(source=small, conf=0.5, verbose=False)[0]

        # Loc ra cac bounding box la "person"
        person_boxes = [
            result for result in results.boxes.data
            if int(result[5]) == PERSON_CLASS_ID
        ]

        # Ve bounding box cho tung nguoi phat hien duoc
        for result in person_boxes:
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = int(x1 * 640 / 208), int(y1 * 480 / 160), int(x2 * 640 / 208), int(y2 * 480 / 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Nhan dien khuon mat da biet
        face_locations, face_names = sfr.detect_known_faces(frame)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if name == "Unknown":
                #luu frame canh bao
                alert_img = "alert_unknown.jpg"
                cv2.imwrite(alert_img, frame)
                send_discord_alert(alert_img)

                # ---- luu khuon mat la vao file ----
                #tao thu muc neu chua ton tai
                unknown_dir = "unknown_faces"
                os.makedirs(unknown_dir, exist_ok=True)

                # cat khuon mat tu khung hinh
                face_img = frame[top:bottom, left:right]

                # dat ten file dua vao thoi gian tranh trung lap
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
                unknown_face_path = os.path.join(unknown_dir, f"unknown_{timestamp}.jpg")

                # rezise lai
                face_img = cv2.resize(face_img, (160, 160))

                #luu anh
                cv2.imwrite(unknown_face_path, face_img)
                print(f"Đã lưu khuôn mặt lạ: {unknown_face_path}")

        with frame_lock:
            processed_frame = frame

# Bat dau luong detection
threading.Thread(target=detection_worker, daemon=True).start()

# Bat dau camera chinh
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)

fps_text = ""
fps_counter = 0
fps_start = time.time()

# Vong lap chinh
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not frame_queue.full():
        frame_queue.put(frame.copy())

    with frame_lock:
        display = processed_frame if processed_frame is not None else frame

    fps_counter += 1
    if time.time() - fps_start >= 1:
        fps_text = f"FPS: {fps_counter}"
        fps_counter = 0
        fps_start = time.time()

    cv2.putText(display, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display, "Nhan phim 'm' de hien camera", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Smart Detection", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # phim ESC
        break
    elif key == ord('m'):
        print("Nhan phim so 0 de chon camera...")

        instruction_display = display.copy()
        cv2.putText(instruction_display, "Nhan Phim 0 de qua camera", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Smart Detection", instruction_display)

        while True:
            key2 = cv2.waitKey(0) & 0xFF
            if ord('0') <= key2 <= ord('9'):
                cam_id = int(chr(key2))
                cap = capture_new_face_by_id(cam_id, cap)
                break
            elif key2 == 27:
                break

cap.release()
cv2.destroyAllWindows()
