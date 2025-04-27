import streamlit as st
from PIL import Image

st.set_page_config(page_title="langchain-streamlit", page_icon="icons/aibot.png")

st.markdown(
    """
    ## A Guidance for langchain-streamlit in various situations
    💬**아래 챗봇 중 하나를 이용해보세요**💬 
    - [**EXAONE-3.5-2.4B-BASELINE**](/EXAONE-3.5-2.4B-BASELINE)
    - [**EXAONE-3.5-2.4B-SFT**](/EXAONE-3.5-2.4B-SFT)
    - [**EXAONE-3.5-2.4B-RAG**](/EXAONE-3.5-2.4B-RAG)
    - [**EXAONE-3.5-7.8B-BASELINE**](/EXAONE-3.5-7.8B-BASELINE)
    """
)