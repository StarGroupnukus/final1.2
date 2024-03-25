import datetime
import json
#from test import find_max_score_object, get_embeddings, save_to_json
import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np

# Загрузка модели и инициализация приложения для анализа лиц
detector = insightface.model_zoo.get_model('./models/model.onnx', download=True)
app = FaceAnalysis()
app.prepare(ctx_id=0)


def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError




# def compare_embeddings(baza_embeddings, list_embeddings, comp_group):
#     results = {}
#     for group, group_data in baza_embeddings.items():
#         print("Group",group, "ComP_group",comp_group)
#         results[group] = {}
#         for compare_embedding in list_embeddings:
#             if group == comp_group:
#
#                 results[group] = {}
#                 for ID, ID_data in group_data.items():
#                     # id_name = int(ID.split('-')[1])
#                     results[group][ID] = {}
#                     for image, embedding in ID_data.items():
#                         if np.any(compare_embedding == 0):
#                             results[group][ID][comp_group] = {'child_id': 0, 'score': 0, 'group_id': comp_group}
#                         else:
#                             # if isinstance(compare_embedding, np.ndarray):
#                             score = detector.compute_sim(np.array(embedding), compare_embedding)
#                             score = round((score * 100), 3)
#                             if score > 1:
#                                 results[group][ID][comp_group] = {"child_id": ID, "score": score,
#                                                                      'group_id': comp_group}
#                             if score == 0:
#                                 results[group][ID][comp_group] = {'child_id': 0, 'score': 0, 'group_id': comp_group}
#     return results
def compare_embeddings(baza_embeddings, list_embeddings, comp_group):
    results = {}
    for group, group_data in baza_embeddings.items():
        print("Group",group, "ComP_group", comp_group)
        results[group] = {}
        for i, compare_embedding in enumerate(list_embeddings):
            if group == comp_group:

                results[group] = {}
                for ID, ID_data in group_data.items():
                    # id_name = int(ID.split('-')[1])
                    results[group][ID] = {}
                    for image, embedding in ID_data.items():
                        if np.any(compare_embedding == 0):
                            results[group][ID][comp_group] = {'child_id': 0, 'score': 0, 'group_id': comp_group,"emb_index":i}
                        else:
                            # if isinstance(compare_embedding, np.ndarray):
                            score = detector.compute_sim(np.array(embedding), compare_embedding)
                            score = round((score * 100), 3)
                            if score > 1:
                                results[group][ID][comp_group] = {"child_id": ID, "score": score,
                                                                     'group_id': comp_group,"emb_index":i}
                            if score == 0:
                                results[group][ID][comp_group] = {'child_id': 0, 'score': 0, 'group_id': comp_group,"emb_index":i}
    return results
def save_to_json(data, file_path):
    try:
        # Открываем файл для записи
        with open(file_path, "w") as file:
            # Записываем значение переменной в файл
            json.dump(data, file, default=convert_numpy, indent=4)
            print(f"Object saved to {file_path}")
    except Exception as e:
        print(f'Не удалось сохранить в {file_path} | Exception -> {e}')




def get_embeddings(file_path):
    try:
        with open(file_path, 'r') as file:
            obj = json.load(file)
            return obj
    except Exception as e:
        print(f"Couldn't read json file ->{e}")


def find_max_score_object(data):
    max_object = None
    max_score = float('-inf')
    for group_data in data.values():
        for image_data in group_data.values():
            for obj in image_data.values():
                if obj['score'] > max_score:
                    max_score = obj['score']
                    max_object = obj
    if max_object['score'] >= 50:
        max_object['status'] = True
    else:
        max_object['status'] = False

    return max_object

