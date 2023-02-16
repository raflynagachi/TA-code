import streamlit as st
import numpy as np
import cv2
from app.algo import helper
from app.algo.dct import *
import zlib
from PIL import Image
from sys import getsizeof

state = {}


def showImage(col, image):
    col.image(image, caption='Uploaded image')


def upload_photo(col, dct):
    uploaded_file = col.file_uploader("Choose an image", type=['png', 'jpeg'])
    if uploaded_file != None:
        image = Image.open(uploaded_file)
        state["stego_image"] = True
        showImage(col, image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dct.set_cover_image(image)
        uploaded_file = None
    else:
        state["stego_image"] = False


def run_extract_message(col1, col2, dct):
    stego_image = None

    # Description
    state["no_extract_process"] = False if (
        state.get("stego_image", True)) else True
    if state.get("stego_image", True):
        col1.write("✔ image uploaded")
    else:
        col1.write("✘ no image")

    # BUTTON
    if col2.button("Extract", disabled=state.get("no_extract_process", True)) and dct.image is not None:
        message = dct.decode(dct.image)  # binary output
        state["message"] = helper.binary_to_bytes(message)
    if col2.button("Uncompress", disabled=(state.get("message", None) is None)):
        comp = zlib.decompress(state["message"])  # bytes output
        state["message"] = comp  # turn bytes to binary
        state["uncompressed"] = True

    if state.get("message", None) is not None:
        # open text file
        text_file = open("data.txt", "w")

        # write string to file
        # print("MESSAGE HERE: \n", state["message"].decode("UTF-8"))
        text_file.write(state["message"].decode("UTF-8"))

        # close file
        text_file.close()
        with open("data.txt", "rb") as file:
            btn = col2.download_button(
                label="Download result",
                data=file,
                file_name="data.txt",
                mime="plain/txt"
            )
