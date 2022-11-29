from utils import input_message, upload_photo
import streamlit as st


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


def body(data):
    st.markdown('---')
    col1, col2 = st.columns(2)
    input_message(col1, data)
    upload_photo(col2, data)
