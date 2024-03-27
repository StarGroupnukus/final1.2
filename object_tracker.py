import json
import os
from pprint import pprint
import cv2
import insightface
from insightface.app import FaceAnalysis
from random import randint
import numpy as np
from datetime import datetime, timedelta
from insightfuncs import find_max_score_object, compare_embeddings, get_embeddings
from funcs import send_report, sort_by_angle

class ObjectTracker:
    def __init__(self, max_age=50, app=None):
        self.objects = {}  # Словарь для хранения объектов
        self.max_age = max_age  # Максимальное количество кадров отсутствия объекта
        self.archive = {}  # Архив для хранения неактивных объектов
        self.app = app
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
            self.objects[obj_id]['data'] = screenshots
            for id, obj in self.archive.items():
                #print(obj.keys())
                obj_embeddings = [obj_one['embedding'] for obj_one in obj['data']]
                sel_embeddings = [obj_one['embedding'] for obj_one in self.objects[obj_id]['data']]
                if self.compare_embeddings(obj_embeddings, sel_embeddings):
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
                #print(self.objects[obj_id])
                is_unique, id = self.check_uniqueness(obj_id, obj_data)
                if is_unique == 2:
                    sorted_obj = sorted(self.objects[obj_id]['data'], key=lambda x: x['det_score'], reverse=True)[:5]
                    self.objects[obj_id]['data'] = sorted_obj
                    self.archive[obj_id] = self.objects[obj_id]
                elif is_unique == 3:
                    sorted_objects = sorted(obj_data['data'], key=lambda x: x['det_score'], reverse=True)[:5]
                    self.archive[id]['data'].extend(sorted_objects)
                    #self.archive[id]['embedding'].extend(obj_data['embedding'])
                to_delete.append(obj_id)
        return to_delete


    def get_five_screenshots_emb(self, obj_data):

        sorted_objects = sort_by_angle(obj_data)
        #print(sorted_objects)
        # Извлечение 5 снимков с наивысшими значениями det_score
        if not sorted_objects:
            return False
        top_5_screenshots = [obj['screenshot'] for obj in sorted_objects]
        face_embeddings = []
        for indx, screenshot in enumerate(top_5_screenshots):
            #cv2.imshow('screenshot', screenshot)
            try:
                face_embedding = self.app.get(screenshot)[0]['embedding']
                obj_data[indx]['embedding'] = face_embedding
                face_embeddings.append(obj_data[indx])
                print('лицо найдено')
            except Exception as e:
                print(f'нет лица| Exception -> {e}')
        if len(face_embeddings) > 0:
            return face_embeddings
        else:
            return False

    def end_video(self, group):
        is_person = False
        self.to_delete(frame_count=12000, max_age=-99999)
        for obj_id, obj_data in self.archive.items():
            send_files = []
            data = obj_data['data']
            embeddings = [obj['embedding'] for obj in data]
            baza_embeddings = get_embeddings('baza.json')
            res = compare_embeddings(baza_embeddings=baza_embeddings, list_embeddings=embeddings, comp_group=group)
            final = find_max_score_object(res)
            score = final['score']
            child_id = final['child_id']
            status = final['status']
            id_known = final['emb_index']
            image = data[id_known]
            filename = f'screenshots/{image["id"]}_{score}_{child_id}_{image["datetime"].strftime("%Y-%m-%d-%H-%M-%S")}.jpg'
            is_person = True
            try:
                # directory = os.path.dirname(filename)
                # if not os.path.exists(directory):
                #     os.makedirs(directory, exist_ok=True)
                # cv2.imwrite(filename, image)
                send_files.append(filename)
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, image['screenshot'])
                send_report(child_id.split('-')[1], send_files, data[0]['datetime'], score, status)
            except Exception as e:
                print(e)

        self.archive.clear()
        return is_person

    def compare_embeddings(self, embeddings1, embeddings2):
        for embeddind1 in embeddings1:
            for embeddind2 in embeddings2:
                if self.detection.compute_sim(embeddind1, embeddind2) > 0.5:
                    return True
        return False

