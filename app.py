import streamlit as st
import glob
from PIL import Image
from classifiers.VIT_classifier import VITClassifier
from classifiers.SVC_classifier import SVCClassifier
from classifiers.CNN_classifier import CNNClassifier
from classifiers.CNN_classifier import BasicBlockNet4
from yolov7_data.yolov7.YOLO_detector import YoloDetector
import os

import sys
sys.path.insert(0, './yolov7_data/yolov7')

st.title(
    'Car Damage Detection'
)

col1, col2 = st.columns(2)

with col1:
    classifier = st.selectbox(
        label='classifier',
        label_visibility='hidden',
        options=['ViT', 'CNN', 'YOLO']
    )

with col2:
    filename = st.file_uploader(
        label='filename',
        label_visibility='hidden',
    )

if classifier == 'YOLO':
    cls = YoloDetector()

elif classifier == 'ViT':
    cls = VITClassifier()

elif classifier == 'CNN':
    cls = BasicBlockNet4()
    cls = CNNClassifier()

else:
    raise Exception

predict_button = st.button('Predict')
st.write("---")

col3, col4 = st.columns(2)
if filename and cls and predict_button:
    with col3:
        st.image(filename)


    with open(f'test_pics/{filename.name}', 'wb') as f:
        f.write(filename.getbuffer())

    if classifier == 'YOLO':
        with col4:
            cls.predict(f'test_pics/{filename.name}')
            st.image('results/image0.jpg')
    else:
        with col4:
            label, score = cls.predict(filename)
            st.subheader(f'Prediction:\t{label}')
            st.subheader(f'Score:\t{score}')

