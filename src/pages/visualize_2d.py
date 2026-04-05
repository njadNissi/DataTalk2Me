import streamlit as st
import plotly.express as px


def render():
    st.title("2D Visualization")
    df = st.session_state.get("data")
    if df is None:
        st.warning("Upload data first")
        return

    cols = df.columns.tolist()
    x = st.selectbox("X-axis", cols)
    y = st.selectbox("Y-axis", cols)
    plot_type = st.selectbox("Plot Type", ["Scatter", "Line", "Histogram"])

    if plot_type == "Scatter":
        fig = px.scatter(df, x=x, y=y)
        fig.update_traces(marker=dict(size=3)) 
    elif plot_type == "Line":
        fig = px.line(df, x=x, y=y)
    else:
        fig = px.histogram(df, x=x)

    st.plotly_chart(fig, use_container_width=True)
