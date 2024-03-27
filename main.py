import os
import time
from datetime import datetime
from insightface.app import FaceAnalysis
from Detection import VideoProcessor


def check_and_select_file(directory_path, group, app, app_emb):
    # Получаем список всех файлов в директории
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Фильтруем список, оставляя только файлы, подходящие под шаблон
    video_files = [f for f in files if f.startswith('output_video_') ]

    # Проверяем, есть ли минимум два таких файла
    if len(video_files) < 2:
        print("В директории недостаточно файлов для обработки.")
        time.sleep(10)
        return

    # Сортируем файлы по дате создания, не включая самый первый
    video_files_sorted = sorted(video_files, key=lambda x: datetime.strptime(x[13:-4], "%Y-%m-%d_%H-%M-%S"))[:-1]
    print(video_files_sorted)
    for video_file in video_files_sorted:

        # Путь к выбранному файлу
        video_file_path = os.path.join(directory_path, video_file)

        print(f"Выбран файл для обработки: {video_file_path}")

        start_time = time.time()
        processor = VideoProcessor(video_url=video_file_path, group=group, app=app, app_emb=app_emb)
        processor.process_video()
        print(time.time() - start_time)
        # Здесь можно добавить логику обработки файла
        # Например: process_file(latest_file_path)


if __name__ == "__main__":
    directory_path = '/home/stargroup/new/camyolo/videos'
    group = 'group-1'
    app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
    app.prepare(ctx_id=0)
    app_emb = FaceAnalysis(providers=['CUDAExecutionProvider'])
    app_emb.prepare(ctx_id=0)
    while True:
        check_and_select_file(directory_path, group, app, app_emb)
