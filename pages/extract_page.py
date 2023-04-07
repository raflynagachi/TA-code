import streamlit as st
from app.view.utils_decode import *
from app.algo.dct import *

st.set_page_config(page_title="Extract Message", page_icon="🔓")

st.markdown("# Extract Message")
st.markdown("""
Extract secret message from an image. Extracted message could be compressed file or plain (text or image).
    
Button:  
**extract** ==> extract message from an stego image  
**decompress** ==> decompressing extracted message using deflate decompression
""")

dct = DCT(is_decode=True)
st.markdown('---')
col1, col2, col3 = st.columns([1, 1, 3])
upload_photo(col3, dct)
run_extract_message(col1, col2, dct)
