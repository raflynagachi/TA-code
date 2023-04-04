import streamlit as st
from app.view.utils_encode import *
from app.algo.dct import *

st.set_page_config(page_title="Embed Message", page_icon="🔒")

st.markdown("# Embed Message")
st.markdown("""
Insert a secret message and cover image. Text messages will be embedded into image using the DCT steganography method. 
Secret messages can be compressed (_optional_) using deflate compression to reduce the size of the message to be embedded.
    
Button:  
**compress** ==> compressing text using deflate compression (_optional_)  
**process** ==> embedding text into image (_image and text are required_)
""")

dct = DCT(is_decode=True)
st.markdown('---')
col1, col2, col3 = st.columns([1, 1, 3])
upload_photo(col3, dct)
input_message(col3, dct)
run_stegano(col1, col2, dct)
