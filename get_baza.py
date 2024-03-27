import os.path
from urllib.request import urlretrieve

from insightface.app import FaceAnalysis
import cv2
from tqdm import tqdm
import numpy as np
from insightfuncs import save_to_json
import requests

app = FaceAnalysis()
app.prepare(ctx_id=0)


def get_dataset(dataset_folder='./baza'):
    try:
        url = 'https://face2.cake-bumer.uz/api/groups_images?kindergarten_id=1'
        response = requests.get(f'{url}/')
        if response.status_code == 200:
            rs = response.json()
            for group in rs['groups']:
                group_id = group.get('group_id')
                group_folder = os.path.join(dataset_folder, f"group-{group_id}")
                children = group.get('children', [])
                # Если список children пустой, пропускаем текущую группу
                if not children:
                    continue
                os.makedirs(group_folder, exist_ok=True)
                # print("group", group_id)
                for child in group.get('children', []):
                    child_id = child.get('child_id')
                    id_folder = os.path.join(dataset_folder, f"group-{group_id}", f"ID-{child_id}")
                    os.makedirs(id_folder, exist_ok=True)
                    # print("ID", child_id)
                    for image in child.get('images', []):
                        image_url = image['url']
                        # полный путь к файлу
                        image = f'https://face2.cake-bumer.uz{image_url}'
                        print(image)
                        image_end = image_url.split('/')[-1]
                        image_name = 'group' + str(group_id) + "_" + "ID" + str(child_id) + "_" + image_end
                        # Сохранение изображения
                        # image_path = os.path.join(f'{dataset_folder}/group-{group_id}/ID-{child_id}', image_name)
                        image_path = os.path.join(f'{dataset_folder}/group-{group_id}/ID-{child_id}', image_end)
                        urlretrieve(image, image_path)

            print("Images successfully saved")
        else:
            print("error", response.status_code)
    except Exception as e:
        return f'Error: {e}'
def data_to_emdeddings(data_folder='baza'):
    names_embeddings_dict = {}
    for group in os.listdir(data_folder):
        group_path = os.path.join(data_folder, group)
        names_embeddings_dict[group] = {}
        if os.path.isdir(group_path):
            person_embeddings = folder_to_embeddings(group_path)
            names_embeddings_dict[group] = person_embeddings
    return names_embeddings_dict

def folder_to_embeddings(folder_path='zips'):
    folder_embeddings_dict = {}
    for dir_name in os.listdir(folder_path):
        dir_path = os.path.join(folder_path, dir_name)
        # Инициализация словаря для каждой группы
        folder_embeddings_dict[dir_name] = {}
        list_embeddings = []
        dirs = os.listdir(dir_path)
        for file in tqdm(dirs):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_file_path = os.path.join(dir_path, file)
                image = cv2.imread(full_file_path)
                try:
                    face_result = app.get(image)[0]
                    face_embedding = face_result['embedding']
                    list_embeddings.append(face_embedding)
                    # Добавление эмбеддингов в словарь
                    folder_embeddings_dict[dir_name][file] = np.stack(list_embeddings[-1], axis=0)
                except Exception as e:
                    folder_embeddings_dict[dir_name][file] = 0
                    # folder_embeddings_dict[dir_name][file] = 0
                    print(f'нет лица в {file}| Exception -> {e}')

        # # Добавление эмбеддингов в словарь
        # for idx, file in enumerate(os.listdir(dir_path)):
        #     if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        #         folder_embeddings_dict[dir_name][file] = np.stack(list_embeddings[idx], axis=0)
    return folder_embeddings_dict

if __name__ == '__main__':
    get_dataset('baza')
    baza_embs = data_to_emdeddings()
    save_to_json(baza_embs, 'baza.json')


