import streamlit as st
import numpy as np
import cv2
from app.algo import helper
from app.algo.dct import *
import zlib
from PIL import Image
from sys import getsizeof
import time

state = {}


def showImage(col, image):
    col.image(image, caption='Uploaded image')


def upload_photo(col, dct):
    upload_file = col.file_uploader(
        "Choose an image", type=['png', 'jpeg', 'jpg'])
    if upload_file != None:
        image = Image.open(upload_file)
        state["stego_image"] = True
        showImage(col, image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        dct.set_cover_image(image)
        upload_file = None
    else:
        state["stego_image"] = False
        state["message"] = None


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
        start_time = time.time()
        message = dct.decode(dct.image)  # binary output
        end_time = time.time()
        state["message"] = helper.binary_to_bytes(message)
        state["uncompressed"] = False
        est_time = end_time - start_time
        col1.write("extract computation time: {:.2f}s".format(est_time))
    if col2.button("Uncompress", disabled=state.get("message", None) is None and state.get("no_extract_process", True) and not state.get("stego_image", False)):
        try:
            start_time = time.time()
            comp = zlib.decompress(state["message"])  # bytes output
            end_time = time.time()
            state["message"] = comp  # turn bytes to binary
            state["uncompressed"] = True
            est_time = end_time - start_time
            col1.write("uncompress computation time: {:.2f}s".format(est_time))
            state['error'] = False
        except Exception as err:
            state['error'] = True
            col1.error("Error: " + str(err), icon="🚨")

    if state.get("message", None) is not None and not state.get("no_extract_process", True):
        # write string to file
        # print("MESSAGE HERE: \n", state["message"].decode("UTF-8"))
        try:
            # open text file
            text_file = open("data.txt", "w")
            extracted_msg = state["message"].decode("UTF-8")
            text_file.write(extracted_msg)

            # close file
            text_file.close()
            with open("data.txt", "rb") as file:
                btn = col2.download_button(
                    label="Download result",
                    data=file,
                    file_name="data.txt",
                    mime="plain/txt"
                )
        except Exception as err:
            state['error'] = True
            col1.error('Error: ' + str(err), icon="🚨")
