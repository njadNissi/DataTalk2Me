import streamlit as st
import pandas as pd

if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(range(1_000_000))

# Add pagination
page_size = 100
page = st.number_input("Page", min_value=1, max_value=len(st.session_state['data'])//page_size + 1, value=1)
start_idx = (page - 1) * page_size
end_idx = start_idx + page_size

# Render only a slice of the data
st.dataframe(st.session_state['data'].iloc[start_idx:end_idx])