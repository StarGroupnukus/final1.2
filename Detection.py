from datetime import datetime
import cv2
import os
import time
from insightface.app import FaceAnalysis
import numpy as np
from sort import Sort
from funcs import convert_detections_to_bbs_format






class VideoProcessor:
    def __init__(self, camera_url, group, app):
        self.camera_url = camera_url
        self.users = []
        self.app = app
        self.group = f'group-{group}_ID'
        self.ids_dict = {}
        self.tracker = Sort(max_age=100, min_hits=8, iou_threshold=0.40)


    def process_video(self):
        cap = cv2.VideoCapture(self.camera_url)
        if not cap.isOpened():
            print("Не удалось подключиться к камере.")
            return

        os.makedirs('screenshots', exist_ok=True)
        number = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Не удалось получить кадр.")
                    break

                frame = cv2.resize(frame, (1280, 720))

                faces = self.app.get(frame)
                #self.save_screenshot(frame, faces, number)
                detections_list = convert_detections_to_bbs_format(faces)


                if len(detections_list) == 0:
                    detections_list = np.empty((0, 5))

                res = self.tracker.update(detections_list)


                boxes_track = res[:, :-1]
                boxes_ids = res[:, -1].astype(int)
                anotation_frame = self.draw_rectangle(frame, faces)
                anotation_frame = self.draw_bounding_boxes_with_id(anotation_frame, boxes_track, boxes_ids)


                cv2.imshow(self.group, anotation_frame)
                number += 1

                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()

    def draw_bounding_boxes_with_id(self, img, bboxes, ids):

        for bbox, id_ in zip(bboxes, ids):
            cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 3)

        return img

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
                score_text = f"Score: {det_score:.2f}"
                cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        return frame
    def save_screenshot(self, frame, faces, number):
        for indx, face in enumerate(faces):
            if True:
                #is_nose_inside(face) and face['det_score'] > 0.70
                ls = [int(value) for value in face['bbox']]
                screenshot = frame[ls[1]-10:ls[3]+10, ls[0]-10:ls[2]+10]
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
    video_url = '/home/stargroup/new/saved_videos/output_video_2024-03-10_11-16-28.avi'
    group = 1

    app = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    processor = VideoProcessor(camera_url=video_url, group=group, app=app)
    processor.process_video()
