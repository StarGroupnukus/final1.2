from datetime import datetime
import cv2
import os
import time
from insightface.app import FaceAnalysis
import numpy as np
from sort import Sort
from funcs import convert_detections_to_bbs_format, add_ids_to_orig_if_matched, filename_to_date, save_video, is_nose_inside
from object_tracker import ObjectTracker
from random import randint



class VideoProcessor:
    def __init__(self, video_url, group, app, app_emb):
        self.video_url = video_url
        self.users = []
        self.app = app_emb
        self.group = group


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
                frame_count += 1

                # if frame_count % 2 != 0:
                #     continue
                frame = cv2.resize(frame, (1280, 720))
                frame_copy = frame.copy()
                faces = self.app.get(frame)



                # anotation_frame = self.draw_rectangle(anotation_frame, faces)
                # cv2.imshow(self.video_url, anotation_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:

            cap.release()
            cv2.destroyAllWindows()


    def draw_rectangle(self, frame, faces):
        for face in faces:
            if True:
                # Рисуем прямоугольник вокруг лица
                x1, y1, x2, y2 = [int(value) for value in face['bbox']]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Отображаем det_score и id
                det_score = face['det_score']
                id = face['id']
                area = face['area']
                score_text = f"{id}_{area}: {det_score:.2f}"
                cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame


if __name__ == "__main__":
    USERNAME = 'admin'
    PASSWORD = 'Babur2001'
    camera_url = f'rtsp://{USERNAME}:{PASSWORD}@192.168.0.119:554/Streaming/Channels/101'
    video_url = '/home/stargroup/new/saved_videos/output_video_2024-03-14_15-18-41.avi'
    group = 'group-1'

    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0)
    app_emb = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app_emb.prepare(ctx_id=0)
    start_time = time.time()
    processor = VideoProcessor(video_url=video_url, group=group, app=app, app_emb=app_emb)
    processor.process_video()
    print(time.time() - start_time)