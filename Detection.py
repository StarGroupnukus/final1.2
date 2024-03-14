from datetime import datetime
import cv2
import os
import time
from insightface.app import FaceAnalysis
import numpy as np
from sort import Sort
from funcs import convert_detections_to_bbs_format, add_ids_to_orig_if_matched, filename_to_date
from object_tracker import ObjectTracker
from random import randint





class VideoProcessor:
    def __init__(self, video_url, group, app, app_emb):
        self.video_url = video_url
        self.users = []
        self.app = app
        self.group = f'group-{group}_ID'
        self.ids_dict = {}
        self.tracker = Sort(max_age=100, min_hits=8, iou_threshold=0.40)
        self.obj_tracker = ObjectTracker(max_age=50, app=app_emb)


    def process_video(self):
        cap = cv2.VideoCapture(self.video_url)
        if not cap.isOpened():
            print("Не удалось подключиться к камере.")
            return

        os.makedirs('screenshots', exist_ok=True)
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось получить кадр.")
                    break

                frame = cv2.resize(frame, (1280, 720))

                faces = self.app.get(frame)
                for face in faces:

                    x1, y1, x2, y2 = [int(value) for value in face['bbox']]
                    screenshot = frame[y1-20:y2+20, x1-20:x2+20]
                    face['screenshot'] = screenshot
                    face['datetime'] = filename_to_date(self.video_url, frame_count)

                detections_list = convert_detections_to_bbs_format(faces)

                if len(detections_list) == 0:
                    detections_list = np.empty((0, 5))

                res = self.tracker.update(detections_list)

                faces = add_ids_to_orig_if_matched(res, faces)

                self.obj_tracker.update(faces, frame_count)
                # anotation_frame = self.draw_rectangle(frame, faces)
                #cv2.imshow(self.group, frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.obj_tracker.end_video()
            cap.release()


    def draw_rectangle(self, frame, faces):
        for face in faces:
            if True:
                # is_nose_inside(face) and face['det_score'] > 0.70
                #name = self.compare_embeddings(self.obj, face['embedding'])
                # Рисуем прямоугольник вокруг лица
                x1, y1, x2, y2 = [int(value) for value in face['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Отрисовываем ключевые точки
                for point in face['kps']:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)  # Красный цвет для ключевых точек

                # Отображаем det_score
                det_score = face['det_score']
                id = face['id']
                score_text = f"{id}_Score: {det_score:.2f}"
                cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame
    def save_screenshot(self, frame, faces, number):
        for indx, face in enumerate(faces):
            if True:
                #is_nose_inside(face) and face['det_score'] > 0.70
                ls = [int(value) for value in face['bbox']]
                screenshot = frame[ls[1]:ls[3], ls[0]:ls[2]]

                now = datetime.now()
                filename = f'screenshots/group-2{number}-{indx}_{now.strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
                if screenshot.size > 0:
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                    cv2.imwrite(filename, screenshot)
                    print(f'Saved {filename}')
                else:
                    print(f"Пустой скриншот для ID, не сохранено.")



if __name__ == "__main__":
    USERNAME = 'admin'
    PASSWORD = 'Babur2001'
    camera_url = f'rtsp://{USERNAME}:{PASSWORD}@192.168.0.119:554/Streaming/Channels/101'
    #video_url = '/home/stargroup/new/saved_videos/output_video_2024-03-10_15-28-47.avi'
    video_url = '/home/stargroup/new/saved_videos/output_video_2024-03-13_16-18-15.avi'
    group = 1

    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0)

    processor = VideoProcessor(video_url=video_url, group=group, app=app)
    processor.process_video()
