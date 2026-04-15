import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import ast


def render():
    st.title("Data Visualization (2D & 3D)")
    
    # --------------------------
    # Data Source Selection
    # --------------------------
    # Build data source options dynamically
    data_options = ['original data']
    if 'preprocessing' in st.session_state and 'data' in st.session_state['preprocessing']:
        data_options.append('preprocessed data')
    data_options.append('my function')
    
    selected_data = st.radio(
        "Select Data Source",
        data_options,
        index=0,
        help="Choose which data to visualize"
    )

    # --------------------------
    # Data Preparation
    # --------------------------
    df = None
    if selected_data == 'original data':
        df = st.session_state.get("data")
        if df is None:
            st.warning("Upload original data first!")
            return
    elif selected_data == 'preprocessed data':
        df = st.session_state['preprocessing']['data']
        if df is None:
            st.warning("Preprocessed data is empty!")
            return

    # --------------------------
    # 2D Visualization Section (Collapsible)
    # --------------------------
    with st.expander("2D Visualization", expanded=True):
        if selected_data != 'my function':
            cols = df.columns.tolist()
            x = st.selectbox("X-axis", cols, key="2d_x")
            y = st.selectbox("Y-axis", cols, key="2d_y")
            plot_type = st.selectbox("Plot Type", ["Scatter", "Line", "Histogram"], key="2d_type")

            # Generate 2D Plot
            if plot_type == "Scatter":
                fig = px.scatter(df, x=x, y=y, title=f"2D Scatter: {x} vs {y}")
                fig.update_traces(marker=dict(size=3))
            elif plot_type == "Line":
                fig = px.line(df, x=x, y=y, title=f"2D Line: {x} vs {y}")
            else:  # Histogram
                fig = px.histogram(df, x=x, title=f"Histogram: {x}")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("2D visualization is not available for custom function mode (3D only)")

    # --------------------------
    # 3D Visualization Section (Collapsible)
    # --------------------------
    with st.expander("3D Visualization", expanded=False):
        if selected_data != 'my function':
            # 3D Plot from Dataset
            cols = df.columns.tolist()
            x_3d = st.selectbox("X", cols, key="3d_x")
            y_3d = st.selectbox("Y", cols, key="3d_y")
            z_3d = st.selectbox("Z", cols, key="3d_z")

            fig_3d = px.scatter_3d(df, x=x_3d, y=y_3d, z=z_3d, 
                                   title=f"3D Scatter: {x_3d}, {y_3d}, {z_3d}")
            fig_3d.update_traces(marker=dict(size=3))
            st.plotly_chart(fig_3d, use_container_width=True)
        
        else:
            # ------------------------------------------
            # 3D Plot from Custom Function (FIXED)
            # ------------------------------------------
            st.subheader("Custom 3D Function Plotter")

            # Persist function in session state
            if "custom_3d_func" not in st.session_state:
                st.session_state["custom_3d_func"] = "np.sin(np.sqrt(x**2 + y**2))"

            func_input = st.text_area(
                "Enter 3D function (z = f(x,y))",
                value=st.session_state["custom_3d_func"],
                height=80,
                help="Use numpy syntax.\nExamples:\n- np.sin(np.sqrt(x**2 + y**2))\n- x**2 + y**2\n- np.cos(x) * np.sin(y)"
            )

            # Save to session state
            st.session_state["custom_3d_func"] = func_input

            x_range = st.slider("X range", -10.0, 10.0, (-5.0, 5.0), step=0.5)
            y_range = st.slider("Y range", -10.0, 10.0, (-5.0, 5.0), step=0.5)
            resolution = st.slider("Resolution", 20, 100, 50)

            # Generate grid
            x = np.linspace(x_range[0], x_range[1], resolution)
            y = np.linspace(y_range[0], y_range[1], resolution)
            X, Y = np.meshgrid(x, y)

            # Safe evaluation
            try:
                # CLEAN SAFE ENVIRONMENT
                allowed_vars = {"np": np, "x": X, "y": Y, "X": X, "Y": Y}
                Z = eval(st.session_state["custom_3d_func"], allowed_vars)

                # Ensure Z is 2D (critical fix)
                if Z.shape != X.shape:
                    st.error("Output shape does not match grid!")
                else:
                    fig_func = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
                    fig_func.update_layout(
                        title=f"z = {st.session_state['custom_3d_func']}",
                        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                        autosize=True
                    )
                    st.plotly_chart(fig_func, use_container_width=True)

            except Exception as e:
                st.error(f"Invalid function: {str(e)}")
                st.info("Try: np.sin(np.sqrt(x**2 + y**2))")


# Run the unified visualization
if __name__ == "__main__":
    render()