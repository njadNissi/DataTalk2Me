import streamlit as st
import time


def temp_show(msg:str, type:str, dur:float):
    placeholder = st.empty()

    if type == 'success':
        placeholder.success(msg)
    elif type == 'info':
        placeholder.info(msg)
    elif type == 'error':
        placeholder.error(msg)
    elif type == 'markdown':
        placeholder.markdown(msg)
    else:
        placeholder.write(msg)

    time.sleep(dur)
    placeholder.empty()