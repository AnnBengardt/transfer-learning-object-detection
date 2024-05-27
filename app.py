import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
from detect import run
import os


def pretrained_yolov5(device="CPU"):
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
    st.text("В перспективе такую модель можно использовать на мусороперерабатывающий заводе для быстрой сортировки или для дронов, собирающих мусор в природных зонах вроде лесов и океанов.")
    st.image("data/metrcis_screenshot.png", caption="Лосс и метрики обученной модели")
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

    option = st.sidebar.radio("Модель", ['Pretrained YOLOv5m', 'Pretrained YOLOv5m - video','Custom dataset YOLOv5s'])

    st.header('Проект для DLS (семестр Осень, 2022): Detection')
    st.subheader("Автор: Марунько Анна")
    st.sidebar.markdown("https://github.com/AnnBengardt/DLS-Detection-Final_project")
    if option == "Pretrained YOLOv5m":
        pretrained_yolov5()
    elif option == "Custom dataset YOLOv5s":
        custom_yolov5s()
    elif option == "Custom dataset YOLOv5s - video":
        video_custom_yolov5s()


if __name__ == '__main__':
    main()


