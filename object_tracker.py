import json
import os
from pprint import pprint
from funcs import default_converter
import cv2
import insightface
from insightface.app import FaceAnalysis
from random import randint
import numpy as np
from datetime import datetime, timedelta

class ObjectTracker:
    def __init__(self, max_age=50):
        self.objects = {}  # Словарь для хранения объектов
        self.max_age = max_age  # Максимальное количество кадров отсутствия объекта
        self.archive = {}  # Архив для хранения неактивных объектов
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0)
        self.detection = insightface.model_zoo.get_model('models/model.onnx', download=True)


    def update(self, new_objects, frame_count):
        # Обновляем существующие объекты
        for obj in new_objects:
            obj_id = obj['id']
            if obj_id in self.objects:
                self.objects[obj_id]['last_seen'] = frame_count
                self.objects[obj_id]['data'].append(obj)
            else:
                self.objects[obj_id] = {'last_seen': frame_count, 'data': [obj]}

        # Удаляем старые объекты

        to_delete = self.to_delete(frame_count, self.max_age)
        for obj_id in to_delete:
            del self.objects[obj_id]

    def check_uniqueness(self, obj_id, obj_data):
        obj_data = obj_data['data']
        screenshots = self.get_five_screenshots_emb(obj_data)
        if screenshots:
            self.objects[obj_id]['embedding'] = screenshots
            for id, obj in self.archive.items():
                print(obj.keys())
                if self.compare_embeddings(obj['embedding'], self.objects[obj_id]['embedding']):
                    print(3, obj_id)
                    return 3, id
            print(2, obj_id)
            return 2, None
        print(1, obj_id)
        return 1, None

    def to_delete(self, frame_count, max_age):
        to_delete = []
        for obj_id, obj_data in self.objects.items():
            if (frame_count - obj_data['last_seen']) > max_age:
                is_unique, id = self.check_uniqueness(obj_id, obj_data)
                if is_unique == 2:
                    sorted_obj = sorted(self.objects[obj_id]['data'], key=lambda x: x['det_score'], reverse=True)[:5]
                    self.objects[obj_id]['data'] = sorted_obj
                    self.archive[obj_id] = self.objects[obj_id]
                elif is_unique == 3:
                    sorted_objects = sorted(obj_data['data'], key=lambda x: x['det_score'], reverse=True)[:5]
                    self.archive[id]['data'].extend(sorted_objects)
                    self.archive[id]['embedding'].extend(obj_data['embedding'])
                to_delete.append(obj_id)
        return to_delete



    def get_five_screenshots_emb(self, obj_data):
        if len(obj_data) < 5:
            return False
        else:
            # Сортировка объектов по det_score в убывающем порядке
            sorted_objects = sorted(obj_data, key=lambda x: x['det_score'], reverse=True)

            # Извлечение 5 снимков с наивысшими значениями det_score
            top_5_screenshots = [obj['screenshot'] for obj in sorted_objects[:5]]
            face_embeddings = []
            for indx, screenshot in enumerate(top_5_screenshots):
                try:

                    face_embedding = self.app.get(screenshot)[0]['embedding']
                    face_embeddings.append(face_embedding)
                    print('лицо найдено')
                except Exception as e:
                    print(f'нет лица| Exception -> {e}')

            if len(face_embeddings) > 0:
                return face_embeddings
            else:
                return False

    def end_video(self):
        self.to_delete(frame_count=12000, max_age=-99999)
        for obj_id, obj_data in self.archive.items():
            data, embeddings = obj_data['data'], obj_data['embedding']
            sort_data = sorted(data, key=lambda x: x['det_score'], reverse=True)[:3]
            for elem in sort_data:
                filename = f'screenshots/{elem["id"]}_{obj_id}_{round(elem["det_score"], 2)}_{elem["datetime"].strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
                print(filename)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, elem['screenshot'])

    def compare_embeddings(self, embeddings1, embeddings2):
        for embeddind1 in embeddings1:
            for embeddind2 in embeddings2:
                if self.detection.compute_sim(embeddind1, embeddind2) > 0.4:
                    return True
        return False

