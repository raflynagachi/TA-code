import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image


def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Algoritma Transform Domain Discrete Cosine Transform (DCT) dan Algoritma Kompresi Lossless Deflate pada Steganografi Citra Digital
        </h1>
    """, unsafe_allow_html=True
                )

    st.caption("""
        <p style='text-align: center; margin-top: 8px;'>
        by Rafly Rigan Nagachi
        </p>
    """, unsafe_allow_html=True
               )

    st.write(
        "Abstrak disini"
    )


def body():
    st.markdown('---')
    upload_photo()


def upload_photo():
    uploaded_file = st.file_uploader("Pilih gambar")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        imageArr = np.array(image)
        st.image(image, caption='Uploaded image')
