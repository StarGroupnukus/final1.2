import numpy as np

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


