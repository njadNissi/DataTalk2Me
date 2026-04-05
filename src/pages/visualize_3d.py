import streamlit as st
import plotly.express as px


def render():
    st.title("3D Visualization")
    df = st.session_state.get("data")
    if df is None:
        st.warning("Upload data first")
        return

    cols = df.columns.tolist()
    x = st.selectbox("X", cols)
    y = st.selectbox("Y", cols)
    z = st.selectbox("Z", cols)

    fig = px.scatter_3d(df, x=x, y=y, z=z)
    fig.update_traces(marker=dict(size=3)) 
    st.plotly_chart(fig, use_container_width=True)