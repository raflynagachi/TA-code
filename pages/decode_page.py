import streamlit as st
from app.view.utils import *
from app.algo.dct import *

st.set_page_config(page_title="Decode Message", page_icon="🔓")

st.markdown("# Decode Message")
st.markdown("""
Extract secret message from an image. Extracted message could be compressed message or plain UTF-8 text.
    
Button:  
**extract** ==> extract message from an stego image  
**uncompressed** ==> uncompressing extracted message using inverse of deflate compression
""")

dct = DCT(decode=True)
st.markdown('---')