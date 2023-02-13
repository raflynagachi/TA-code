import streamlit as st
from app.view.utils import *
from app.algo.dct import *

st.set_page_config(page_title="Encode Message", page_icon="🔒")

st.markdown("# Encode Message")
st.markdown("""
Insert a secret message in the form of text into an image. Text messages will be embedded into images using the DCT steganography method
Secret messages can be compressed (_optional_) to reduce the size of the message to be embedded.
    
Button:  
**compress** ==> compressing text using deflate compression (_optional_)  
**process** ==> embedding text into image (_image and text are required_)
""")

dct = DCT(decode=True)
st.markdown('---')
col1, col2, col3 = st.columns([1, 1, 3])
upload_photo(col3, dct)
input_message(col3, dct)
run_stegano(col1, col2, dct)