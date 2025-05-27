import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self):
        # Danh sách chứa encoding và tên tương ứng của các khuôn mặt đã biết
        self.known_face_encodings = []
        self.known_face_names = []

        # Tỉ lệ resize khung hình để tăng tốc nhận diện (giảm độ phân giải)
        self.frame_resizing = 0.5

    def load_encoding_images(self, images_path):
        """
        Tải và encode tất cả hình ảnh trong thư mục chỉ định.
        Mỗi hình ảnh được giả định là chứa một khuôn mặt và tên file là tên của người đó.
        """
        # Lấy tất cả đường dẫn hình ảnh trong thư mục
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"Đã tìm thấy {len(images_path)} ảnh cần encode.")

        # Xóa dữ liệu cũ nếu có
        self.known_face_encodings.clear()
        self.known_face_names.clear()

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không thể đọc ảnh: {img_path}")
                continue

            # Chuyển ảnh từ BGR (OpenCV) sang RGB (face_recognition)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Lấy encoding của khuôn mặt (nếu có)
            encodings = face_recognition.face_encodings(rgb_img)
            if len(encodings) == 0:
                print(f"Không tìm thấy khuôn mặt trong ảnh: {img_path}. Bỏ qua ảnh này.")
                continue

            img_encoding = encodings[0]

            # Lấy tên từ file (không lấy phần đuôi như .jpg, .png)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
            print(f"Đã thêm khuôn mặt: {filename}")

        print("Hoàn tất việc nạp encoding từ thư mục ảnh.")

    def detect_known_faces(self, frame):
        """
        Tìm và nhận diện các khuôn mặt đã biết trong khung hình đầu vào.
        Trả về vị trí khuôn mặt và tên tương ứng (nếu có).
        """
        # Resize nhỏ khung hình để tăng tốc
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Phát hiện vị trí các khuôn mặt trong ảnh đã resize
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # Tính khoảng cách giữa khuôn mặt này với tất cả các khuôn mặt đã biết
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Nếu khoảng cách đủ nhỏ (tức là giống nhau), gán tên người
            threshold = 0.45
            if face_distances[best_match_index] < threshold:
                name = self.known_face_names[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

        # Chuyển lại tọa độ khuôn mặt về kích thước gốc
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
