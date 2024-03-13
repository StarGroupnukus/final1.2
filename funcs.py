import numpy as np
import re
from datetime import datetime, timedelta


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


def add_ids_to_orig_if_matched(sort, orig):
    matched_orig = []  # Список для хранения совпавших элементов с ID

    for i, det in enumerate(sort):
        det_center = [(det[0] + det[2]) / 2, (det[1] + det[3]) / 2]  # Центр детекции из sort

        best_match_idx = -1
        best_distance = float('inf')

        for j, o in enumerate(orig):
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

def save_screenshots(frames, time):
    pass

def default_converter(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Преобразование массивов NumPy в списки
    elif isinstance(obj, np.float32):
        return float(obj)  # Преобразование float32 в стандартный float
    elif isinstance(obj, datetime):
        return obj.isoformat()  # Преобразование datetime в строку ISO 8601
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")