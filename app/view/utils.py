import streamlit as st
import numpy as np
from PIL import Image


def input_message(col, data):
    message = col.text_input('Pesan rahasia')
    data.setMessage(message)


def showImage(col, image):
    col.image(image, caption='Uploaded image')


def upload_photo(col, data):
    uploaded_file = col.file_uploader("Pilih gambar")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        data.setImage(image)
        # imageArr = np.array(image)
        # data.setImageArr(imageArr)
        showImage(col, image)
