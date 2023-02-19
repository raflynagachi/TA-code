import streamlit as st
import numpy as np
import cv2
from app.algo import helper
import zlib
from PIL import Image
from sys import getsizeof

state = {}


def input_message(col, dct):
    message = col.text_input('Pesan rahasia')
    if message != "":
        state["no_compress"] = False
        state["compressed"] = False
        # state["message"] = str.encode(message)
        dct.set_message(str.encode(message))
    else:
        state["no_compress"] = True
        # state["message"] = ""
        dct.set_message("")


def showImage(col, image):
    col.image(image, caption='Uploaded image')


def upload_photo(col, dct):
    uploaded_file = col.file_uploader("Choose an image", type=['png', 'jpeg'])
    if uploaded_file != None:
        image = Image.open(uploaded_file)
        state["image"] = True
        showImage(col, image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dct.set_cover_image(image)
        uploaded_file = None
    else:
        state["image"] = False


def run_stegano(col1, col2, dct):
    print("STATE: ", state)
    stego_image = None
    # BUTTON
    if col2.button("Compress", disabled=state.get("no_compress", True)):
        print("\n\nmasuk\n\n")
        comp = zlib.compress(dct.message)
        state["message"] = comp
        state["compressed"] = True
    if col2.button("Process", disabled=state.get("no_process", True)) and dct.channel != None:
        try:
            msg = state["message"] if state.get(
                "message", b'') != b'' else dct.message
            stego_image = dct.encode(helper.bytes_to_binary(msg))
            state["processed"] = True
        except Exception as err:
            st.error('error: ' + str(err), icon="🚨")
    if state.get("processed", True) and stego_image is not None:
        image = cv2.imread("stego_image.png", flags=cv2.IMREAD_COLOR)
        col1.write("PSNR: " + str(helper.PSNR(dct.ori_img, image)))
        with open("stego_image.png", "rb") as file:
            btn = col2.download_button(
                label="Download result",
                data=file,
                file_name="stego_image.png",
                mime="image/png"
            )

    # Description
    state["no_process"] = False if (
        state.get("image", True)) else True
    if state.get("image", True) and dct.image is not None:
        col1.write("✔ image uploaded " + "{}".format(dct.image.shape[:2]))
    else:
        col1.write("✘ no image")

    if dct.message != "":
        col1.write("✔ message uploaded")
        col1.write("Text size: " +
                   "{:.2f}KB".format(getsizeof(dct.message)/1000))
        if state["no_compress"] == False and state["compressed"] == True:
            col1.write("Text size (compressed): " +
                       "{:.2f}KB".format(getsizeof(state["message"])/1000))
            col1.write("Compression ratio: " + "{:.2f}".format(getsizeof(dct.message) /
                                                               getsizeof(state["message"])))
    else:
        col1.write("✘ no message")
