# src/pages/data_plotting.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import ast
from typing import List, Optional, Tuple
import warnings
import io
from io import BytesIO
import base64
import zipfile

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")

# Set page config (only if this is the main entry point)
def setup_page():
    try:
        st.set_page_config(
            page_title="Advanced Data Visualization",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        pass  # Already set by main app

def get_color_palette(n: int) -> List[str]:
    """Generate distinct color palette for multiple traces"""
    base_colors = px.colors.qualitative.Plotly
    if n <= len(base_colors):
        return base_colors[:n]
    # Generate extended palette using HSL if needed
    return [f'hsl({i * 360 / n}, 70%, 50%)' for i in range(n)]

def validate_data_columns(df, columns: List[str]) -> Tuple[bool, List[str]]:
    """Validate selected columns exist in dataframe"""
    missing = [col for col in columns if col not in df.columns]
    return len(missing) == 0, missing

def render_2d_visualization(df):
    """Enhanced 2D Visualization Component - FIXED: X vs MULTIPLE Y"""
    cols = df.columns.tolist()
    
    # Layout (unchanged)
    col1, col2, col3, col4, col5 = st.columns([2, 3, 1, 2, 1])
    
    with col1:
        # X axis: SINGLE column (time, recommended)
        x_column = st.selectbox(
            "X-axis (independent var.)",
            cols,
            key="2d_x",
            help="Select one column for X-axis (e.g. time)",
            index=0 if cols else 0
        )
    
    with col2:
        # Y axis: MULTIPLE series (val, increaser, decreaser)
        y_columns = st.multiselect(
            "Y-axis (multiple dependent values allowed)",
            cols,
            key="2d_y",
            help="Select one or more targets for Y-axis",
            default=cols[1:2] if len(cols)>1 else None
        )
    
    with col3:
        plot_type = st.selectbox(
            "Plot Type",
            ["Scatter", "Line", "Histogram", "Bar", "Box Plot"],
            key="2d_type"
        )
    
    with col4:
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7, 0.1, key="2d_opacity")

    with col5:
        show_grid = st.checkbox("Show Grid", True, key="2d_grid")
    
    # Validation
    if not y_columns:
        st.warning("⚠️ Please select at least one Y-axis series")
        return
    
    is_valid, missing_cols = validate_data_columns(df, [x_column] + y_columns)
    if not is_valid:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        return

    # Plot: X (fixed) vs MULTIPLE Y series
    try:
        fig = go.Figure()
        colors = get_color_palette(len(y_columns))
        
        for i, y_col in enumerate(y_columns):
            color = colors[i]
            
            if plot_type == "Scatter":
                fig.add_trace(go.Scatter(
                    x=df[x_column],
                    y=df[y_col],
                    mode='markers',
                    name=y_col,
                    marker=dict(size=4, color=color, opacity=opacity),
                    hovertemplate=f"{x_column}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Line":
                fig.add_trace(go.Scatter(
                    x=df[x_column],
                    y=df[y_col],
                    mode='lines+markers',
                    name=y_col,
                    line=dict(color=color, width=2),
                    marker=dict(size=3, opacity=opacity),
                    hovertemplate=f"{x_column}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Histogram":
                fig.add_trace(go.Histogram(
                    x=df[y_col],
                    name=y_col,
                    opacity=opacity,
                    marker=dict(color=color),
                ))
            
            elif plot_type == "Bar":
                fig.add_trace(go.Bar(
                    x=df[x_column],
                    y=df[y_col],
                    name=y_col,
                    opacity=opacity,
                    marker=dict(color=color),
                    hovertemplate=f"{x_column}: %{{x}}<br>{y_col}: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Box Plot":
                fig.add_trace(go.Box(
                    y=df[y_col],
                    name=y_col,
                    marker=dict(color=color),
                    opacity=opacity,
                ))

        # Layout
        fig.update_layout(
            title=f"{plot_type} | {x_column} vs {', '.join(y_columns)}",
            xaxis_title=x_column,
            yaxis_title="Value",
            legend_title="Y Series",
            hovermode="closest",
            template="plotly_white", # other options: "plotly_dark", "ggplot2", "seaborn"
            height=600,
            showlegend=True,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid)
        )

        if plot_type == "Histogram":
            fig.update_layout(barmode='overlay')

        st.plotly_chart(fig, width='stretch')

        # ========== SIMPLE, FAST, CLEAN DOWNLOAD ==========
        st.markdown("---")
        st.subheader("📦 Download All Formats (PNG, SVG, PDF, JPEG, HTML)")
        generic_path = "data-talk2me_2d_plot"

        if st.button("💾 Download Plots Full Package", width="stretch", key="dld2dbtn"):
            try:
                # Show loading (prevents browser freeze warning)
                with st.spinner("Preparing your files..."):
                    status_placeholder = st.empty()
                    zip_buffer = io.BytesIO()

                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                        # Step 1
                        status_placeholder.info("🔄 Exporting PNG...")
                        zf.writestr(f"{generic_path}.png", fig.to_image(format="png", scale=2))

                        # Step 2
                        status_placeholder.info("🔄 Exporting JPEG...")
                        zf.writestr(f"{generic_path}.jpeg", fig.to_image(format="jpeg", scale=2))
                        
                        # Step 3
                        status_placeholder.info("🔄 Exporting SVG...")
                        zf.writestr(f"{generic_path}.svg", fig.to_image(format="svg"))
                        
                        # Step 4
                        status_placeholder.info("🔄 Exporting PDF...")
                        zf.writestr(f"{generic_path}.pdf", fig.to_image(format="pdf"))
                        
                        # Step 5
                        status_placeholder.info("🔄 Generating interactive HTML...")
                        html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
                        zf.writestr(f"{generic_path}.html", html_str.encode("utf-8"))
                        
                    status_placeholder.success("✅ All formats ready! Download below:")
                    zip_buffer.seek(0)

                    # ✅ TRIGGER DOWNLOAD
                    st.download_button(
                        label="✅ Click to Download ZIP",
                        data=zip_buffer,
                        file_name=f"{generic_path}s.zip",
                        mime="application/zip",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"Error: {str(e)}")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def render_3d_visualization(df, selected_data: str):
    """Enhanced 3D Visualization Component - FIXED: X vs MULTIPLE Y vs Z + No Duplicate Keys"""
    if selected_data != 'my function':
        cols = df.columns.tolist()
        
        col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
        
        with col1:
            # X = SINGLE column (time)
            x_3d_column = st.selectbox(
                "X-axis (Usually independent variable)",
                cols,
                key="3d_x",
                help="Select ONE column for X-axis (e.g. time)",
                index=0 if cols else 0
            )
        
        with col2:
            # Y = MULTIPLE series
            y_3d_columns = st.multiselect(
                "Y-axis (Usually dependent variables, multiple allowed)",
                cols,
                key="3d_y",
                help="Select one or more targets for Y-axis",
                default=cols[1:2] if len(cols)>1 else None
            )
        
        with col3:
            use_const_z = st.checkbox("Use Constant Z Value", value=len(cols) < 3)
            # ✅ FIXED: MOVE CONSTANT Z INPUT OUTSIDE LOOP (NO DUPLICATE KEY)
            z_value = 0.0
            if use_const_z:
                z_value = st.number_input(
                    "Constant Z Value",
                    value=0.0,
                    step=0.1,
                    key="const_z",
                    help="Fixed Z coordinate for all points"
                )
            # Z axis select
            if not use_const_z:
                z_3d_column = st.selectbox(
                    "Z-axis Feature",
                    cols,
                    key="3d_z",
                    help="Select feature for Z-axis",
                    index=2 if len(cols) > 2 else 0
                )
        
        with col4:
            st.markdown("### 3D Plot Settings")
            marker_size = st.slider("Marker Size", 1, 10, 3, 1)
            opacity = st.slider("Opacity", 0.1, 1.0, 0.8, 0.1)
            show_legend = st.checkbox("Show Legend", True)

        # Validation
        if not y_3d_columns:
            st.warning("⚠️ Please select at least one Y-axis series")
            return
        
        all_selected = [x_3d_column] + y_3d_columns
        if not use_const_z:
            all_selected.append(z_3d_column)
        
        is_valid, missing_cols = validate_data_columns(df, all_selected)
        if not is_valid:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            return

        # Generate 3D Plot
        try:
            fig_3d = go.Figure()
            colors = get_color_palette(len(y_3d_columns))

            # Loop over MULTIPLE Y series
            for i, y_col in enumerate(y_3d_columns):
                # Prepare Z data
                if use_const_z:
                    z_data = np.full(len(df), z_value)
                    z_label = f"Z = {z_value}"
                else:
                    z_data = df[z_3d_column]
                    z_label = z_3d_column

                # 3D Trace
                fig_3d.add_trace(go.Scatter3d(
                    x=df[x_3d_column],
                    y=df[y_col],
                    z=z_data,
                    mode='markers',
                    name=y_col,
                    marker=dict(
                        size=marker_size,
                        color=colors[i % len(colors)],
                        opacity=opacity,
                        line=dict(width=0.5, color='white')
                    ),
                    hovertemplate=(
                        f"{x_3d_column}: %{{x}}<br>"
                        f"{y_col}: %{{y}}<br>"
                        f"{z_label}: %{{z}}<extra></extra>"
                    )
                ))

            # Layout
            fig_3d.update_layout(
                title=f"3D Scatter Plot | {x_3d_column} vs {', '.join(y_3d_columns)} vs {z_label}",
                scene=dict(
                    xaxis_title=x_3d_column,
                    yaxis_title="Y Series",
                    zaxis_title=z_label,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    zaxis=dict(showgrid=True)
                ),
                legend=dict(
                    title="Y Series",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.02,
                    visible=show_legend
                ),
                height=700,
                template="plotly_white"
            )

            st.plotly_chart(fig_3d, width='stretch')

            # ========== SIMPLE, FAST, CLEAN DOWNLOAD ==========
            st.markdown("---")
            st.subheader("📦 Download All Formats (PNG, SVG, PDF, JPEG, HTML)")
            generic_path = "data-talk2me_3d_plot"

            if st.button("💾 Download Plots Full Package", width="stretch", key="dld3dbtn"):
                try:
                    # Show loading (prevents browser freeze warning)
                    with st.spinner("Preparing your files..."):
                        status_placeholder = st.empty()
                        zip_buffer = io.BytesIO()

                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                            # Step 1
                            status_placeholder.info("🔄 Exporting PNG...")
                            zf.writestr(f"{generic_path}.png", fig_3d.to_image(format="png", scale=2))

                            # Step 2
                            status_placeholder.info("🔄 Exporting JPEG...")
                            zf.writestr(f"{generic_path}.jpeg", fig_3d.to_image(format="jpeg", scale=2))
                            
                            # Step 3
                            status_placeholder.info("🔄 Exporting SVG...")
                            zf.writestr(f"{generic_path}.svg", fig_3d.to_image(format="svg"))
                            
                            # Step 4
                            status_placeholder.info("🔄 Exporting PDF...")
                            zf.writestr(f"{generic_path}.pdf", fig_3d.to_image(format="pdf"))
                            
                            # Step 5
                            status_placeholder.info("🔄 Generating interactive HTML...")
                            html_str = fig_3d.to_html(full_html=True, include_plotlyjs="cdn")
                            zf.writestr(f"{generic_path}.html", html_str.encode("utf-8"))
                            
                        status_placeholder.success("✅ All formats ready! Download below:")
                        zip_buffer.seek(0)

                        # ✅ TRIGGER DOWNLOAD
                        st.download_button(
                            label="✅ Click to Download ZIP",
                            data=zip_buffer,
                            file_name=f"{generic_path}s.zip",
                            mime="application/zip",
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"Error: {str(e)}")

        except Exception as e:
            st.error(f"❌ Error generating 3D plot: {str(e)}")
            st.exception(e)
    
    else:
        # Custom 3D Function Plotter (UNCHANGED)
        st.subheader("🎨 Custom 3D Function Plotter")
        
        example_functions = {
            "Sphere": "np.sqrt(1 - (x**2 + y**2)/100)",
            "Sine Wave": "np.sin(np.sqrt(x**2 + y**2))",
            "Paraboloid": "x**2 + y**2",
            "Cosine Product": "np.cos(x) * np.sin(y)"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            func_input = st.text_area(
                "Enter 3D function (z = f(x,y))",
                value=st.session_state.get("custom_3d_func", "np.sin(np.sqrt(x**2 + y**2))"),
                height=80,
                help="Use numpy syntax (np = numpy, x/y = grid coordinates)",
                key="custom_func_input"
            )
        with col2:
            st.markdown("### Examples")
            selected_example = st.selectbox("Quick Load", list(example_functions.keys()), key="func_example")
            if st.button("Load Example"):
                st.session_state["custom_3d_func"] = example_functions[selected_example]
                st.rerun()

        st.session_state["custom_3d_func"] = func_input

        with st.expander("Advanced Settings", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                x_range = st.slider("X range", -20.0, 20.0, (-5.0, 5.0), step=0.5, key="x_range")
            with col2:
                y_range = st.slider("Y range", -20.0, 20.0, (-5.0, 5.0), step=0.5, key="y_range")
            with col3:
                resolution = st.slider("Resolution", 20, 200, 50, key="resolution")
            
            surface_color = st.color_picker("Surface Base Color", "#636efa", key="surface_color")
            show_contours = st.checkbox("Show Contours", True, key="show_contours")

        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        try:
            safe_env = {
                "np": np,
                "x": X,
                "y": Y,
                "X": X,
                "Y": Y,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sqrt": np.sqrt,
                "exp": np.exp,
                "log": np.log,
                "abs": np.abs,
                "pow": np.pow
            }
            
            Z = eval(func_input, {"__builtins__": None}, safe_env)

            if Z.shape != X.shape:
                st.error("❌ Function output shape doesn't match grid dimensions!")
                st.info(f"Expected shape: {X.shape}, Got: {Z.shape}")
            else:
                fig_func = go.Figure()
                
                surface_kwargs = {
                    "z": Z,
                    "x": X,
                    "y": Y,
                    "colorscale": [[0, surface_color], [1, px.colors.sequential.Viridis[-1]]],
                    "opacity": 0.8,
                    "name": "Function Surface"
                }
                
                if show_contours:
                    surface_kwargs["contours"] = {
                        "z": {"show": True, "usecolormap": True, "highlightcolor": "white", "project_z": True}
                    }
                
                fig_func.add_trace(go.Surface(**surface_kwargs))

                fig_func.update_layout(
                    title=f"3D Surface: z = {func_input}",
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z",
                        xaxis=dict(showgrid=True),
                        yaxis=dict(showgrid=True),
                        zaxis=dict(showgrid=True)
                    ),
                    height=700,
                    template="plotly_white",
                    hovermode="closest"
                )

                st.plotly_chart(fig_func, width='stretch')

        except SyntaxError as e:
            st.error(f"❌ Syntax Error: {str(e)}")
            st.info("Check your numpy syntax (e.g., use np.sin() instead of sin())")
        except NameError as e:
            st.error(f"❌ Name Error: {str(e)}")
            st.info("Only numpy functions (np.*) are allowed. Use np.sin(), np.cos(), etc.")
        except Exception as e:
            st.error(f"❌ Error evaluating function: {str(e)}")
            st.info("Try a simpler function first: np.sin(np.sqrt(x**2 + y**2))")

def render():
    """Main rendering function for data visualization page"""
    setup_page()
    
    st.title("📊 Advanced Data Visualization (2D & 3D)")
    st.divider()

    # --------------------------
    # Data Source Selection
    # --------------------------
    st.subheader("Data Source", anchor="data-source")
    with st.expander("Select which data to visualize", expanded=True):
    
        # Build data source options dynamically with better feedback
        data_options = []
        data_status = {}
        
        if 'data' in st.session_state and st.session_state['data'] is not None and not st.session_state['data'].empty:
            data_options.append('original data')
            data_status['original data'] = f"{len(st.session_state['data'])} rows and {len(st.session_state['data'].columns)} columns"
        else:
            data_status['original data'] = "❌ No data uploaded"
        
        if 'preprocessing' in st.session_state and 'data' in st.session_state['preprocessing'] and not st.session_state['preprocessing']['data'].empty:
            data_options.append('preprocessed data')
            data_status['preprocessed data'] = f"✅ {len(st.session_state['preprocessing']['data'])} rows, {len(st.session_state['preprocessing']['data'].columns)} columns"
        else:
            data_status['preprocessed data'] = "❌ Preprocessed data empty"
        
        data_options.append('my function')
        data_status['my function'] = "✏️ Custom 3D function plotter"

        # Show data source status
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_data = st.radio(
                "Select Data Source",
                data_options,
                index=0 if data_options else 0,
                help="Choose which data to visualize",
                key="data_source"
            )
        with col2:
            st.markdown(f"## {data_status[selected_data]}")

    # --------------------------
    # Data Preparation
    # --------------------------
    df = None
    if selected_data == 'original data':
        df = st.session_state.get("data")
        if df is None or df.empty:
            st.warning("⚠️ Please upload original data first (via Data Upload page)")
            return
    elif selected_data == 'preprocessed data':
        df = st.session_state['preprocessing']['data']
        if df is None or df.empty:
            st.warning("⚠️ Preprocessed data is empty! Please run preprocessing first.")
            return

    # --------------------------
    # Visualization Sections
    # --------------------------
    # 2D Visualization (only for dataset sources)
    if selected_data != 'my function':
        with st.expander("2D Visualization", expanded=True):
            render_2d_visualization(df)
    
    # 3D Visualization (all sources)
    with st.expander("3D Visualization", expanded=False):
        render_3d_visualization(df, selected_data)

# Run the visualization
if __name__ == "__main__":
    render()