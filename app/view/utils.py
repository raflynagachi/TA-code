import streamlit as st
import numpy as np
import cv2
from app.algo import helper
import zlib
from PIL import Image
from sys import getsizeof


def input_message(col, dct):
    message = col.text_input('Pesan rahasia')
    if message != "":
        st.session_state["no_compress"] = False
        st.session_state["compressed"] = False
        st.session_state["message"] = str.encode(message)
        dct.set_message(helper.bytes_to_binary(str.encode(message)))
    else:
        st.session_state["no_compress"] = True
        st.session_state["message"] = ""
        dct.set_message("")


def showImage(col, image):
    col.image(image, caption='Uploaded image')


def upload_photo(col, dct):
    uploaded_file = col.file_uploader("Pilih gambar")
    if uploaded_file != None:
        image = Image.open(uploaded_file)
        st.session_state["image"] = True
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dct.set_cover_image(image)
    else:
        st.session_state["image"] = False


def display_information(col, dct):
    pass


def run_stegano(col1, col2, dct):
    stego_image = None
    # BUTTON
    if col2.button("Compress", disabled=st.session_state.get("no_compress", True)) and dct.message != "":
        comp = zlib.compress(helper.binary_to_bytes(dct.message))
        comp = helper.bytes_to_binary(comp)
        dct.set_message(comp)
        st.session_state["compressed"] = True
    if col2.button("Process", disabled=st.session_state.get("no_process", True)) and dct.channel != None and dct.message != "":
        st.session_state["processed"] = True
        stego_image = dct.encode(dct.message)
    if st.session_state.get("processed", True) and stego_image is not None:
      with open("stego_image.png", "rb") as file:
          btn = col2.download_button(
                  label="Download result",
                  data=file,
                  file_name="stego_image.png",
                  mime="image/png"
                )

    # Description
    st.session_state["no_process"] = False if (
        st.session_state.get("image", True) and dct.message != None) else True
    if st.session_state.get("image", True):
        col1.write("✔ image uploaded " + "{}".format(dct.image.shape[:2]))
    else:
        col1.write("✘ no image")

    if dct.message != "" and st.session_state["message"] != "":
        col1.write("✔ message uploaded")
        col1.write("Text size: " +
                   "{:.2f}KB".format(getsizeof(st.session_state["message"])/1000))
        if st.session_state["no_compress"] == False and st.session_state["compressed"] == True:
            col1.write("Text size (compressed): " +
                       "{:.2f}KB".format(getsizeof(dct.message)/1000))
            col1.write("Compression ratio: " + "{:.2f}".format(getsizeof(st.session_state["message"]) /
                                                               getsizeof(dct.message)))
    else:
        col1.write("✘ no message")
    
