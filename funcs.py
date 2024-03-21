import math
import os
import shutil
import mimetypes

import cv2
import numpy as np
import re
from datetime import datetime, timedelta

import requests


def convert_detections_to_bbs_format(detections):
    """
    Конвертирует список словарей обнаружений в формат bbs,
    ожидаемый deep_sort_realtime. Каждая детекция преобразуется в кортеж,
    состоящий из списка [left, top, width, height], confidence и заглушки для detection_class.

    Параметры:
        detections (list): Список словарей с обнаружениями, где каждый словарь содержит
                           ключ 'bbox' с координатами в формате numpy array и
                           'det_score' с уверенностью обнаружения.

    Возвращает:
        list: Список кортежей в формате ([left, top, width, height], confidence, detection_class).
    """
    bbs = []
    for detection in detections:
        bbox = detection['bbox']
        confidence = detection['det_score']

        # Добавление кортежа в итоговый список
        bbs.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
    return np.array(bbs)

def is_nose_inside(face):
    if face["det_score"] >= 0.50:
        # Получаем ключевые точки
        kps = face['kps']
        bbox = face['bbox']
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        x_left_eye, x_right_eye, y_right_eye = kps[0][0], kps[1][0], kps[1][1]
        x_nose, y_nose = kps[2]
        check = x_left_eye < x_nose < x_right_eye

        return check, area
    return False, 0



def save_video(filename):
    shutil.copy(filename, '/home/stargroup/new/saved_videos')


def add_ids_to_orig_if_matched(sort, orig):
    matched_orig = []  # Список для хранения совпавших элементов с ID

    for i, det in enumerate(sort):
        det_center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]  # Центр детекции из sort

        best_match_idx = -1
        best_distance = float('inf')

        for j, o in enumerate(orig):
            if 'screenshot' not in o:
                continue  # Пропустить объекты без ключа "screenshot"

            o_center = [(o['bbox'][0] + o['bbox'][2]) / 2, (o['bbox'][1] + o['bbox'][3]) / 2]  # Центр bbox из orig

            # Рассчитываем Евклидово расстояние между центрами
            distance = np.sqrt((det_center[0] - o_center[0]) ** 2 + (det_center[1] - o_center[1]) ** 2)

            if distance < best_distance:
                best_distance = distance
                best_match_idx = j

        # Если нашли подходящее совпадение, добавляем к нему ID
        if best_match_idx != -1 and det[-1] > 0:  # Убедимся, что ID существует и больше 0
            orig_with_id = orig[best_match_idx].copy()  # Копируем, чтобы не изменять оригинальный список
            orig_with_id['id'] = int(det[-1])
            matched_orig.append(orig_with_id)

    return matched_orig


def filename_to_date(filename, frame_count):
    # Используем регулярное выражение для поиска даты и времени в имени файла
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})", filename)
    if match:
        # Группы захвата используются для извлечения компонентов даты и времени
        year, month, day, hour, minute, second = match.groups()
        # Преобразуем извлеченные строки в целые числа
        year, month, day, hour, minute, second = map(int, [year, month, day, hour, minute, second])
        # Создаем объект datetime из извлеченных значений
        datetime_obj = datetime(year, month, day, hour, minute, second) + timedelta(seconds=int(frame_count/20))
        return datetime_obj

def save_screenshots(frames, id, det_score):
    filename = f"baza_screenshots/{str(id)}_{round(det_score, 2)}.jpg"
    cv2.imwrite(filename, frames)
    print('Saved screenshot', filename)
def send_report(id, file_paths, time, score, status):

    url = 'https://face2.cake-bumer.uz/api/reports2'
    data = {
        'group_id': '1',
        'child_id': str(id),
        'score': str(score),
        'time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': '1' if status else '0',
    }
    files_data = []

    try:
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            mime_type, _ = mimetypes.guess_type(file_name)
            files_data.append(('images[]', (file_name, open(file_path, 'rb'), mime_type)))
        files = tuple(files_data)

        response = requests.post(url, data=data, files=files, headers={'Accept': 'application/json'})
        print(response.status_code, response.text)
    except Exception as e:
        print(e)
    finally:
        for _, (filename, file, _) in files_data:
            file.close()

if __name__ == '__main__':
    send_report(1, ['screenshots/1_1_88_2024-03-15-17-26-53.jpg', 'screenshots/1_1_88_2024-03-15-17-26-53.jpg'], datetime.now(), 0.59)