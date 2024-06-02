import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
from detect import run
import os


def pretrained_yolov5(device="CPU"):
        with st.expander("Выбранная архитектура - YOLOv5"):
                st.subheader("YOLOv5 (You Only Look Once version 5) — это популярная модель для задач детектирования объектов в изображениях.")
                st.write("---")
                st.subheader("Архитектура модели:")
                st.write("""Backbone: Основная сеть (backbone) используется для извлечения признаков из входного изображения.""")
                st.write("""Neck: Эта часть предназначена для объединения и обработки признаков, извлечённых на различных уровнях глубины сети.""")
                st.write("""Head: Последняя часть архитектуры, которая отвечает за предсказание координат боксов, классов объектов и уверенности в детекции.""")
                st.write("---")
                st.subheader("Ключевые особенности YOLOv5:")
                st.write("""End-to-End обучение: Модель обучается и делает предсказания за один проход через сеть, что делает её быстрой и эффективной.""")
                st.write("""Мультискалярное предсказание: Использование разных масштабов для предсказаний помогает обнаруживать объекты разных размеров.""")
                st.write("""Встроенные механизмы аугментации данных: YOLOv5 включает методы аугментации данных для улучшения общей способности модели к генерализации.""")
                st.write("---")
                st.subheader("Пайплайн работы с YOLOv5:")
                st.write("""Предобработка изображений: Масштабирование и нормализация изображений.""")
                st.write("""Пропуск через Backbone: Извлечение признаков на различных уровнях глубины.""")
                st.write("""Обработка в Neck: Объединение признаков для усиления детекции.""")
                st.write("""Предсказание в Head: Генерация боксов, классов и уверенности.""")
                st.image("data/yolo.jpg")
        with st.expander("Результаты работы модели"):
                st.image("data/pic1.png")
                st.image("data/pic2.png")
                st.image("data/pic3.png")
                st.write("""В целом, результаты работы базовой архитектуры неплохие. Особенно хорошо у модели получаются сцены с людьми, транспортом и некоторыми животными (как на примере, с котами, собаками и коровами, а вот лошадь получилась не очень). Тяжело придумать какой-либо путь улучшения, когда уже по сути всё сделали за тебя и выпустили уже далеко не одну вариацию YOLO после использованной тут YOLOv5. """)
                st.write("""Так что есть два простых варианта:
* Можно брать модель побольше. Я встроила именно YOLOv5m для скорости инференса и уменьшения используемых ресурсов на хостинге, но в принципе YOLOv5l тоже должен быстро и не очень затратно работать. Так как это более глубокая версия модели, то результаты должны улучшиться.
* И, конечно, можно взять более новую версию YOLO. Например, примерно год назад вышла YOLOv8, которая сейчас регулярно обновляется.""")
        image_file = st.file_uploader("Загрузить изображение:", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Загруженное', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts) + image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            # call Model prediction--
            model = torch.hub.load("ultralytics/yolov5", "yolov5m", pretrained=True)
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            # --Display predicton

            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Результат', use_column_width='always')


def custom_yolov5s(device="CPU"):
    st.header('Обученная YOLOv5s на кастомном датасете')
    st.subheader("Датасет: виды перерабатываемого и неперерабатываемого мусора")
    st.image("data/metrcis_screenshot.png", caption="Лосс и метрики обученной модели")
    with st.expander("Подробнее о пайплайне обучения"):
                st.write("""YOLOv5s была дообучена на кастомном датасете. Был использован аугментированный и размеченный датасет с сервиса Roboflow с перерабатываемым и неперерабатываемым видами мусора - https://universe.roboflow.com/thanakon21010-gmail-com/waste-gf9v2/dataset/4 . Применённые аугментации:
* Flip: Horizontal, Vertical
* 90° Rotate: Clockwise, Counter-Clockwise, Upside Down
* Crop: 0% Minimum Zoom, 14% Maximum Zoom
* Rotation: Between -17° and +17°
* Shear: ±20° Horizontal, ±12° Vertical
* Grayscale: Apply to 25% of images
* Hue: Between -29° and +29°
* Saturation: Between -36% and +36%
* Brightness: Between -28% and +28%
* Exposure: Between -25% and +25%
* Blur: Up to 2px
* Noise: Up to 3% of pixels""")
                st.write("""Классы: 
                - Battery Hazardous waste
                - Bottle Recycle waste
                - Glass Bottle Recycle waste
                - Can Recycle waste
                - Leaf Compostable
                - Mama General waste
                - Mask Hazardous waste
                - Milo Recycle waste
                - Ovaltine Recycle waste
                - Pen Hazardous waste""")
                st.write("""В целом, модель работает хорошо, но не все классы хорошо детектируются. Также иногда наблюдаются проблемы с баундинг боксами (либо один объект определяется двумя баундинг боксами, либо границы баундинг бокса некорректны). Но это всё равно неплохой результат для маленькой версии YOLOv5s и не очень долгого цикла обучения (170 эпох) относительно базовой версии модели (500 эпох).""")
                st.write("---")        
                st.write("""Были переопределены параметры модели в yaml-конфигурационном файле, а также переиспользован скрипт обучения от создателей модели""")
                st.image("data/model.jpg") 
    imgpath = glob.glob('data/images/*')
    imgsel = st.slider('Выбрать случайную картинку из тестовой выборки', min_value=1, max_value=len(imgpath), step=1)
    image_file = imgpath[imgsel - 1]
    submit = st.button("Начать детекцию")
    col1, col2 = st.columns(2)
    with col1:
        img = Image.open(image_file)
        st.image(img, caption='Выбранное изображение', use_column_width='always')
    with col2:
        if image_file is not None and submit:
            # call Model prediction--
            model = torch.hub.load("ultralytics/yolov5", "custom", path='data/models/yoloTrained.pt', force_reload=True)
            pred = model(image_file)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                # --Display predicton
                img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                st.image(img_, caption='Результат')


def video_custom_yolov5s():
    st.header('Обученная YOLOv5s на кастомном датасете')
    st.subheader("Датасет: виды перерабатываемого и неперерабатываемого мусора")
    st.subheader("Тест на видео: желательно загружать видео с мусором")
    uploaded_video = st.file_uploader("Загрузить видео", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:
        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts) + uploaded_video.name)
        outputpath = os.path.join('data/outputs', os.path.basename(imgpath))

        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Загруженное видео")
        run(weights='data/models/yoloTrained.pt', source=imgpath, device="cpu")
        st_video2 = open(outputpath, 'rb')
        video_bytes2 = st_video2.read()
        st.video(video_bytes2)
        st.write("Результат")


def main():

    option = st.sidebar.radio("Модель", ['Pretrained YOLOv5m','Custom dataset YOLOv5s'])

    st.header('Трансферное обучение для решения задач Object Detection на примере выбранной базовой архитектуры')
    st.subheader("Автор: Марунько Анна МО23-1м")
    st.sidebar.markdown("https://github.com/AnnBengardt/transfer-learning-object-detection")
    if option == "Pretrained YOLOv5m":
        pretrained_yolov5()
    elif option == "Custom dataset YOLOv5s":
        custom_yolov5s()
    #elif option == "Custom dataset YOLOv5s - video":
       # video_custom_yolov5s()


if __name__ == '__main__':
    main()


