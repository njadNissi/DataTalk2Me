# src/pages/data_plotting.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import ast
from typing import List, Optional, Tuple
import warnings

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
    """Enhanced 2D Visualization Component"""
    cols = df.columns.tolist()
    
    # Layout with better spacing
    col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
    
    with col1:
        x_columns = st.multiselect(
            "X-axis Features", 
            cols, 
            key="2d_x",
            help="Select one or more features for X-axis",
            default=cols[0] if cols else None
        )
    
    with col2:
        y_column = st.selectbox(
            "Y-axis (Target)", 
            cols, 
            key="2d_y",
            help="Select target feature for Y-axis",
            index=1 if len(cols) > 1 else 0
        )
    
    with col3:
        plot_type = st.selectbox(
            "Plot Type", 
            ["Scatter", "Line", "Histogram", "Bar", "Box Plot"], 
            key="2d_type"
        )
    
    with col4:
        st.markdown("### Settings")
        opacity = st.slider("Opacity", 0.1, 1.0, 0.7, 0.1, key="2d_opacity")
        show_grid = st.checkbox("Show Grid", True, key="2d_grid")

    # Validation
    if not x_columns:
        st.warning("⚠️ Please select at least one X-axis feature")
        return
    
    is_valid, missing_cols = validate_data_columns(df, x_columns + [y_column])
    if not is_valid:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        return

    # Generate plot
    try:
        fig = go.Figure()
        colors = get_color_palette(len(x_columns))
        
        for i, x_col in enumerate(x_columns):
            color = colors[i]
            
            if plot_type == "Scatter":
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_column],
                    mode='markers',
                    name=x_col,
                    marker=dict(size=4, color=color, opacity=opacity),
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_column}: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Line":
                fig.add_trace(go.Scatter(
                    x=df[x_col],
                    y=df[y_column],
                    mode='lines+markers',
                    name=x_col,
                    line=dict(color=color, width=2),
                    marker=dict(size=3, opacity=opacity),
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_column}: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Histogram":
                fig.add_trace(go.Histogram(
                    x=df[x_col],
                    name=x_col,
                    opacity=opacity,
                    marker=dict(color=color),
                    hovertemplate=f"Value: %{{x}}<br>Count: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Bar":
                fig.add_trace(go.Bar(
                    x=df[x_col],
                    y=df[y_column],
                    name=x_col,
                    opacity=opacity,
                    marker=dict(color=color),
                    hovertemplate=f"{x_col}: %{{x}}<br>{y_column}: %{{y}}<extra></extra>"
                ))
            
            elif plot_type == "Box Plot":
                fig.add_trace(go.Box(
                    y=df[x_col],
                    name=x_col,
                    marker=dict(color=color),
                    opacity=opacity,
                    hovertemplate=f"Feature: {x_col}<br>Value: %{{y}}<extra></extra>"
                ))

        # Layout configuration
        fig.update_layout(
            title=f"{plot_type} Plot: {', '.join(x_columns)} vs {y_column}",
            xaxis_title="X Features" if len(x_columns) > 1 else x_columns[0],
            yaxis_title=y_column if plot_type != "Box Plot" else "Value",
            legend_title="Features",
            hovermode="closest",
            template="plotly_white",
            height=600,
            showlegend=True,
            xaxis=dict(showgrid=show_grid),
            yaxis=dict(showgrid=show_grid)
        )
        
        # Special layout for histograms
        if plot_type == "Histogram":
            fig.update_layout(barmode='overlay')
            fig.update_traces(xbins=dict(size=(df[x_col].max() - df[x_col].min())/50))
        
        # Special layout for box plots
        if plot_type == "Box Plot":
            fig.update_layout(xaxis_title="Features", yaxis_title="Value")

        # Render plot with responsive width
        st.plotly_chart(fig, width='stretch')

    except Exception as e:
        st.error(f"❌ Error generating plot: {str(e)}")
        st.exception(e)  # For debugging (can be removed in production)

def render_3d_visualization(df, selected_data: str):
    """Enhanced 3D Visualization Component"""
    if selected_data != 'my function':
        # 3D Plot from Dataset
        cols = df.columns.tolist()
        
        # Better layout for 3D controls
        col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])
        
        with col1:
            x_3d_columns = st.multiselect(
                "X-axis Features",
                cols,
                key="3d_x",
                help="Select one or more features for X-axis",
                default=cols[0] if cols else None
            )
        
        with col2:
            y_3d_columns = st.multiselect(
                "Y-axis Features",
                cols,
                key="3d_y",
                help="Select one or more features for Y-axis",
                default=cols[1] if len(cols) > 1 else None
            )
        
        with col3:
            use_const_z = st.checkbox("Use Constant Z Value", value=len(cols) < 3)
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
        if not x_3d_columns or not y_3d_columns:
            st.warning("⚠️ Please select at least one X and one Y feature")
            return
        
        # Validate columns
        all_selected = x_3d_columns + y_3d_columns
        if not use_const_z:
            all_selected.append(z_3d_column)
        
        is_valid, missing_cols = validate_data_columns(df, all_selected)
        if not is_valid:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
            return

        # Generate 3D Plot
        try:
            fig_3d = go.Figure()
            colors = get_color_palette(len(x_3d_columns) * len(y_3d_columns))
            trace_count = 0

            for i, x_col in enumerate(x_3d_columns):
                for j, y_col in enumerate(y_3d_columns):
                    # Skip identical axis combinations
                    if x_col == y_col and not use_const_z and z_3d_column == x_col:
                        continue
                    
                    # Prepare Z data
                    if use_const_z:
                        z_value = st.number_input(
                            "Constant Z Value",
                            value=0.0,
                            step=0.1,
                            key="const_z",
                            help="Fixed Z coordinate for all points"
                        )
                        z_data = np.full(len(df), z_value)
                        z_label = f"Z = {z_value}"
                    else:
                        z_data = df[z_3d_column]
                        z_label = z_3d_column

                    # Trace name
                    if len(x_3d_columns) > 1 and len(y_3d_columns) > 1:
                        trace_name = f"{x_col} vs {y_col}"
                    elif len(x_3d_columns) > 1:
                        trace_name = x_col
                    else:
                        trace_name = y_col

                    # Add 3D scatter trace
                    fig_3d.add_trace(go.Scatter3d(
                        x=df[x_col],
                        y=df[y_col],
                        z=z_data,
                        mode='markers',
                        name=trace_name,
                        marker=dict(
                            size=marker_size,
                            color=colors[trace_count % len(colors)],
                            opacity=opacity,
                            line=dict(width=0.5, color='white')
                        ),
                        hovertemplate=(
                            f"{x_col}: %{{x}}<br>"
                            f"{y_col}: %{{y}}<br>"
                            f"{z_label}: %{{z}}<extra></extra>"
                        )
                    ))
                    trace_count += 1

            # Update layout
            fig_3d.update_layout(
                title=f"3D Scatter Plot: {', '.join(x_3d_columns)} vs {', '.join(y_3d_columns)} vs {z_label}",
                scene=dict(
                    xaxis_title="X Features",
                    yaxis_title="Y Features",
                    zaxis_title=z_label,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True),
                    zaxis=dict(showgrid=True)
                ),
                legend=dict(
                    title="Feature Combinations",
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

        except Exception as e:
            st.error(f"❌ Error generating 3D plot: {str(e)}")
            st.exception(e)
    
    else:
        # Enhanced Custom 3D Function Plotter
        st.subheader("🎨 Custom 3D Function Plotter")
        
        # Function input with examples
        example_functions = {
            "Sphere": "np.sqrt(1 - (x**2 + y**2)/100)",
            "Sine Wave": "np.sin(np.sqrt(x**2 + y**2))",
            "Paraboloid": "x**2 + y**2",
            "Cosine Product": "np.cos(x) * np.sin(y)"
        }
        
        # Quick select examples
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

        # Save to session state
        st.session_state["custom_3d_func"] = func_input

        # Advanced controls
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

        # Generate grid
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)

        # Safe function evaluation
        try:
            # Restrict to safe numpy functions only
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
            
            # Evaluate function safely
            Z = eval(func_input, {"__builtins__": None}, safe_env)

            # Validate output shape
            if Z.shape != X.shape:
                st.error("❌ Function output shape doesn't match grid dimensions!")
                st.info(f"Expected shape: {X.shape}, Got: {Z.shape}")
            else:
                # Create surface plot
                fig_func = go.Figure()
                
                # Add surface trace
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

                # Update layout
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
    
    # Build data source options dynamically with better feedback
    data_options = []
    data_status = {}
    
    if 'data' in st.session_state and st.session_state['data'] is not None and not st.session_state['data'].empty:
        data_options.append('original data')
        data_status['original data'] = f"✅ {len(st.session_state['data'])} rows, {len(st.session_state['data'].columns)} columns"
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
        st.markdown(f"**Status:** {data_status[selected_data]}")

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
    st.divider()
    
    # 2D Visualization (only for dataset sources)
    if selected_data != 'my function':
        with st.expander("2D Visualization", expanded=True):
            render_2d_visualization(df)
    
    # 3D Visualization (all sources)
    st.divider()
    with st.expander("3D Visualization", expanded=True):
        render_3d_visualization(df, selected_data)

    # --------------------------
    # Additional Features
    # --------------------------
    st.divider()
    with st.expander("📋 Additional Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Last Plot as PNG", key="download_png"):
                # In a real app, you would save the last plot to a BytesIO buffer
                st.success("Plot download started (feature would implement plotly's write_image)")
        
        with col2:
            if st.button("📄 Download Last Plot as HTML", key="download_html"):
                st.success("Interactive plot download started (feature would implement plotly's write_html)")
        
        with col3:
            dark_mode = st.checkbox("🌙 Dark Mode for Plots", key="dark_mode")
            if dark_mode:
                # Apply dark theme (would need to update plot templates)
                st.info("Dark mode would switch plot template to 'plotly_dark'")

# Run the visualization
if __name__ == "__main__":
    render()