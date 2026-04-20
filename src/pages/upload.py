import streamlit as st
import pandas as pd
import numpy as np
import os
from src.core import feature_analysis as fan, dataset_processing as dsp, utils as utils
import ast 
from src.core.image_lab import * 

# =========================================================
# 🔐 STATE INITIALIZATION
# =========================================================
def init_state():
    if "data" not in st.session_state:
        st.session_state["data"] = None

    if "history" not in st.session_state:
        st.session_state["history"] = [] # list of tuples ('change name', df_copy)

    if "data_version" not in st.session_state:
        st.session_state["data_version"] = 0

    if "column_version" not in st.session_state:
        st.session_state["column_version"] = 0
        
    if "imglab" not in st.session_state:
        st.session_state["imglab"] = {}


def retouch_history(keep_last: int = 2):
    """
        Replace old history DataFrames with their .head() to save memory
        Keep only the last 'keep_last' entries as full DataFrames
    """
    if "history" not in st.session_state:
        return

    full_history = st.session_state["history"]
    new_history = []

    # Iterate all history items
    for i, (label, df) in enumerate(full_history):
        # Keep recent entries FULL
        if i >= len(full_history) - keep_last:
            new_history.append((label, df.copy()))
        # Replace old entries with .head()
        else:
            new_history.append((label, df.head().copy()))

    # Replace history
    st.session_state["history"] = new_history
    
# =========================================================
# 🔄 CENTRAL UPDATE FUNCTION
# =========================================================
def update_data(df):
    st.session_state["data"] = df.copy()
    st.session_state["data_size"] = len(df)
    st.session_state["data_version"] += 1
    st.session_state["column_version"] += 1

    retouch_history()
    

# =========================================================
# 🎯 MAIN APP
# =========================================================
def render():
    init_state()

    st.title("🧾 Data Lab")

    upload_tab, edit_tab, explore_tab, imglab_tab, hist_tab = st.tabs(["📂 Upload", "✏️ Edit", "📊 Explore", "🖼️ Image Studio", "🧾 History"])

    # =========================================================
    # 📂 UPLOAD
    # =========================================================
    with upload_tab:
        uploaded_file = st.file_uploader("Upload your csv file", type=["csv"])

        st.subheader("📌 Example Datasets")

        # Create data folder if it doesn't exist (safe)
        if not os.path.exists("data"):
            os.makedirs("data")

        # Get all CSV files in data/ folder
        csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]

        if csv_files:
            cols = st.columns(len(csv_files))
            for i, file in enumerate(csv_files):
                with cols[i]:
                    if st.button(f"📄 {file}"):
                        file_path = os.path.join("data", file)
                        df = pd.read_csv(file_path)
                        update_data(df)
                        st.session_state["history"] = [("Loaded Example Dataset: " + file, df.copy())]
                        st.rerun()  # refresh to show data
        else:
            st.info("No CSV files found in the /data folder.")

        # --------------------------
        # UPLOADED / EXAMPLE DATA LOGIC
        # --------------------------
        df = st.session_state.get("data")

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state["uploaded_file_name"] = uploaded_file.name
            update_data(df)
            st.session_state["history"] = [("Uploaded File", df.copy())]

        if df is not None:
            st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)} | Sample dataset")
            sample_df = dsp.get_representative_sample(df, target_col=None, sample_size=150)
            st.dataframe(sample_df, height=600)

            # RESET BUTTON
            if st.button("🗑️ Release file"):
                init_state()
                st.session_state["data"] = None
                st.success("✅ File released")
                st.rerun()
        else:
            if not uploaded_file and len(csv_files) > 0:
                st.info("👆 Upload a CSV or select an example dataset")
            
    # =========================================================
    # ✏️ EDIT
    # =========================================================
    with edit_tab:
        # -----------------------------
        # CREATE DATASET
        # -----------------------------
        df = st.session_state.get("data")
        if df is None:
            st.warning("No dataset loaded")

            n_rows = st.number_input("Number of rows", min_value=1, value=10)

            try:
                if st.button("➕ Create Dataset"):
                    df = pd.DataFrame({
                        "x0": ["0"]*n_rows,
                        "x1": ["0"]*n_rows,
                        "y": ["0"]*n_rows
                    })

                    update_data(df)
                    st.session_state["history"] = [("Created Dataset", df.copy())]
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")

        else:
            # -----------------------------
            # DATA EDITOR
            # -----------------------------
            st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")
            editor_key = f"editor_{st.session_state.column_version}_{len(st.session_state.data.columns)}"
            edited_df = st.data_editor(
                st.session_state["data"],
                num_rows="dynamic",
                width="stretch",
                key=editor_key
            )

            if not edited_df.equals(st.session_state["data"]):
                st.session_state["history"].append(("Edited Data", st.session_state["data"].copy()))
                update_data(edited_df)

            # Get non-numeric columns
            categorical_cols = st.session_state["data"].select_dtypes(exclude=['number']).columns.tolist()
            if categorical_cols:
                st.markdown("---")
                st.subheader("🔡 Encode Categorical Columns to Numbers")

                c1, c2, c3, c4 = st.columns([2, 4, 2, 2])
                with c1:
                    select_all = st.checkbox("Select All Categorical Columns")
                    if select_all:
                        selected_cols = categorical_cols  # auto-select all
                    else:
                        with c2:
                            selected_cols = st.multiselect(
                                "Select column(s) to encode",
                                categorical_cols
                            )
                with c3:
                    if st.button("Encode Column to Numerical Values"):
                        temp_df = st.session_state["data"].copy()

                        # Encode every selected column
                        for col_name in selected_cols:
                            encoded_col, le = fan.encode_labels(temp_df[col_name].astype(str))
                            temp_df[col_name] = encoded_col
                            st.session_state[f"{col_name}_encoder"] = le  # Store encoder for potential inverse transform

                        # Save history + update data
                        encoded_names = ", ".join(selected_cols)
                        update_data(temp_df)
                        st.session_state["history"].append((f"Encoded: {encoded_names}", st.session_state["data"].copy()))
                        with c4:
                            utils.temp_show(f"✅ Successfully encoded: **{encoded_names}**", 'success', 1)

                        st.rerun()
            st.markdown("---")
            # -----------------------------
            # COLUMN OPERATIONS
            # -----------------------------
            col1, col2 = st.columns(2)

            with col1:
                new_col = st.text_input("New column name")
                if st.button("➕ Add Column") and new_col:
                    df_new = st.session_state["data"].copy()
                    df_new[new_col] = np.nan  # default value

                    st.session_state["history"].append((f"Added {new_col} column", df_new.copy()))
                    update_data(df_new)
                    st.rerun()

            with col2:
                del_col = st.selectbox(
                    "Delete column",
                    [""] + list(st.session_state["data"].columns)
                )

                if st.button("🗑️ Delete Column") and del_col:
                    df_new = st.session_state["data"].copy()
                    df_new = df_new.drop(columns=[del_col])

                    st.session_state["history"].append((f"Deleted {del_col} column", df_new.copy()))
                    update_data(df_new)
                    st.rerun()

            # =====================================================
            # ⚡ PYTHON COLUMN GENERATOR
            # =====================================================
            st.markdown("---")
            st.subheader("⚡ Python Column Generator")

            pycol1, pycol2 = st.columns([9, 1])
            st.info("Available columns: " + ", ".join(df.columns))
            with pycol1:
                previous_code = st.session_state.get("python_script", "")
                user_code = st.text_area("Python 3.x interpreter",
                    height=120, 
                    help="[1] Can use multiline code\n[2] Use prefix 'col_' in column name to create new columns):\n\tn = len(x0)\n\tcol_x0 = np.arange(-5,5)\n\tcol_x1 = np.random.randint(-10,15)\n\tcol_x2 = x0 - x1**2"
                )

            with pycol2:
                if st.button("▶️ Run Code"):
                    try:
                        # Validate syntax safely
                        tree = ast.parse(user_code)
                        utils.temp_show("✅ Valid Python syntax", 'success', .5)
                        
                        df_temp = df.copy()
                        local_env = {col: df_temp[col].values for col in df_temp.columns}
                        global_env = {
                            "__builtins__": {},
                            "np": np,
                            "len": len,
                            "min": min,
                            "max": max,
                            "sum": sum,
                            'list': list,
                            'dict': dict,
                            'int': int,
                            'float': float,
                            'str': str,
                            'bool': bool
                        }
                        exec(user_code, global_env, local_env)

                        for key, value in local_env.items():
                            if not key.startswith("col_"):
                                continue  # Only process variables starting with col_
                            
                            col_name = key.replace("col_", "") # Only process variables named col_...
                     
                            # Convert to numpy array (safe handling)
                            if isinstance(value, (list, np.ndarray)):
                                arr = np.array(value)

                                try:
                                    arr = arr.astype(np.int64)
                                except:
                                    arr, le = fan.encode_labels(arr.astype(str))
                                    st.session_state[f"{col_name}_encoder"] = le  # Store encoder for potential inverse transform

                                    # show the changes
                                    clean_classes = [str(cls) for cls in le.classes_]
                                    mapping = dict(zip(clean_classes, range(len(le.classes_))))
                                    st.session_state["class_names_mapping"] = mapping.copy()
                                    st.session_state["history"].append((f"Encoded column '{col_name}' with mapping: {mapping}", df_temp.copy()))
                                    st.caption(f"Encoded '{col_name}' with mapping: {mapping}.")
                                
                                df_temp[col_name] = arr

                            # 🔴 FIX 2: Length match check
                            if len(value) != len(df_temp):
                                st.warning(f"⚠️ '{col_name}' length mismatch — skipped")
                                continue

                        update_data(df_temp)
                        st.session_state["history"].append((f"Generated columns [{', '.join(key[4:] for key in local_env if key.startswith('col_'))}] via code", df.copy()))
                        st.session_state['python_script'] = user_code
                        st.rerun()

                    except SyntaxError as e:
                        st.error(f"❌ Syntax error: {e}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

                # -----------------------------
                # UNDO
                # -----------------------------
                if st.button("↩️ Undo"):
                    if len(st.session_state["history"]) > 1:
                        st.session_state["history"].pop()
                        prev_change, prev_df = st.session_state["history"][-1]
                        update_data(prev_df)
                        st.rerun()
                    else:
                        st.warning("No more undo steps")

    # =========================================================
    # 📊 EXPLORE
    # =========================================================
    with explore_tab:
        df = st.session_state.get("data")

        if df is None:
            st.warning("No dataset available")
        else:
            st.subheader("Dataset Overview")
            st.write(f"Shape: {df.shape}")

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

            st.write("🔢 Numeric:", numeric_cols)
            st.write("🔤 Categorical:", categorical_cols)

            st.subheader("Missing Values")
            missing = df.isnull().sum()
            st.dataframe(missing[missing > 0])

            st.subheader("Statistics")
            st.dataframe(df.describe())

            st.subheader("Correlation Matrix")
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                st.dataframe(corr)
            else:
                st.info("Need at least 2 numeric columns for correlation matrix")

            st.subheader("Preprocessing")

            if st.button("Fill NA with Mean"):
                df_new = df.copy()
                df_new[numeric_cols] = df_new[numeric_cols].fillna(df_new[numeric_cols].mean())
                update_data(df_new)
                st.rerun()

            st.info("Changes apply globally.")

    # =========================================================
    # 🖼️ IMAGE STUDIO (TAB 5)
    # =========================================================
    with imglab_tab:
        st.title("🎨 AI-Powered Image Studio")
        st.caption("Professional-grade image editing — simple enough for everyone")
        # --------------------------
        # STATE INITIALIZATION FOR IMAGES
        # --------------------------
        if "original_image" not in st.session_state['imglab']:
            st.session_state['imglab']["original_image"] = None
        if "edited_image" not in st.session_state['imglab']:
            st.session_state['imglab']["edited_image"] = None
        if "crop_coords" not in st.session_state['imglab']:
            st.session_state['imglab']["crop_coords"] = (0, 0, 0, 0)
        if "history" not in st.session_state['imglab']:
            st.session_state['imglab']["history"] = []
        # --------------------------
        # UPLOAD SECTION
        # --------------------------
        up1_col, eg_col = st.columns([5, 5])
        with up1_col:
            uploaded_image = st.file_uploader(
                "Upload JPG/PNG/WEBP/HEIC",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="image_uploader"
            )

        # Example images (like your CSV example datasets)
        with eg_col:
            st.subheader("📸 Example Images")
            example_images = {
                # "Nature": "https://images.unsplash.com/photo-1501854140801-50d01698950b?w=800",
                "Portrait": "https://images.unsplash.com/photo-1499996860823-5214fcc65f8f?w=800",
                # "Cityscape": "https://images.unsplash.com/photo-1543349689-9a4d426bee8e?w=800"
            }

            col_ex = st.columns(len(example_images))
            for i, (name, url) in enumerate(example_images.items()):
                with col_ex[i]:
                    if st.button(f"🖼️ {name}"):
                        try:
                            import requests, io
                            response = requests.get(url)
                            img = Image.open(io.BytesIO(response.content)).convert("RGB")
                            st.session_state['imglab']["original_image"] = img
                            st.session_state['imglab']["edited_image"] = img.copy()
                            st.session_state['imglab']['history'].append((f'`{name}` example image loaded', img.copy()))
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load example: {e}")

        # --------------------------
        # MAIN EDITING WORKFLOW
        # --------------------------
        if uploaded_image:
            try:
                img = Image.open(uploaded_image).convert("RGB")
                st.session_state['imglab']["original_image"] = img
                st.session_state['imglab']["edited_image"] = img.copy()
            except Exception as e:
                st.error(f"❌ Failed to load image: {e}")

        # Get current images from state
        original_img = st.session_state['imglab'].get("original_image")
        edited_img = st.session_state['imglab'].get("edited_image")

        if original_img is None:
            st.info("👉 Upload an image or select an example to start editing!")
        else:
            # --------------------------
            # LAYOUT: SIDEBAR (CONTROLS) + MAIN (PREVIEW)
            # --------------------------
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.subheader("🛠️ Edit Controls")
                # --------------------------
                # BASIC EDITS EXPANDER     #
                # --------------------------
                with st.expander("🔧 Basic Edits", expanded=True):
                    # CROP TOOL
                    st.subheader("✂️ Crop")
                    st.caption("Click & drag on preview to select crop area")
                    crop_aspect = st.selectbox(
                        "Aspect Ratio", ["No crop", "Free Crop", "1:1 (Square)", "4:3", "16:9", "9:16"]
                    )                # Convert selection to aspect ratio NUMBER
                    aspect_ratio = None
                    enable_cropper = True
                    if crop_aspect == "No crop":
                        enable_cropper = False
                    if crop_aspect == "1:1 (Square)":
                        aspect_ratio = (1, 1)
                    elif crop_aspect == "4:3":
                        aspect_ratio = (4, 3)
                    elif crop_aspect == "16:9":
                        aspect_ratio = (16, 9)
                    elif crop_aspect == "9:16":
                        aspect_ratio = (9, 16)
                    else:
                        aspect_ratio = None  # Free

                    if st.button("🖼️ Apply cropping"):
                        st.session_state['history'].append((f'Image cropped with aspect: {crop_aspect}', edited_img.copy()))

                    # Resize
                    st.subheader("📏 Resize")
                    width = st.number_input("Width (px)", min_value=100, value=edited_img.width)
                    height = st.number_input("Height (px)", min_value=100, value=edited_img.height)
                    if st.button("🔄 Resize Image"):
                        edited_img = edited_img.resize((width, height), Image.Resampling.LANCZOS)
                        st.session_state['imglab']["edited_image"] = edited_img
                        st.session_state['imglab']['history'].append((f'Image resized to ({width}x{height})', edited_img.copy()))
                        st.rerun()
                    
                    # Rotate/Flip
                    st.subheader("🔄 Rotate/Flip")
                    rotate_deg = st.slider("Rotate (degrees)", 0, 360, 0)
                    flip_horizontal = st.checkbox("Flip Horizontal")
                    flip_vertical = st.checkbox("Flip Vertical")
                
                    if st.button("✅ Apply Rotate/Flip"):
                        edited_img = edited_img.rotate(rotate_deg, expand=True)
                        if flip_horizontal:
                            edited_img = ImageOps.mirror(edited_img)
                            st.session_state['imglab']['history'].append(('Image horizontally flipped', edited_img.copy()))
                        if flip_vertical:
                            edited_img = ImageOps.flip(edited_img)
                            st.session_state['history'].append(('Image vertically flipped', edited_img.copy()))
                        st.session_state['imglab']["edited_image"] = edited_img
                        st.rerun()
                    
                    # Reset basic edits
                    if st.button("🔄 Reset Basic Edits", type="secondary"):
                        st.session_state['imglab']["edited_image"] = original_img.copy()
                        st.session_state['imglab']['history'] = [(f'`{name}` example image loaded', img.copy())]
                        st.rerun()

                # --------------------------
                # ENHANCEMENT EXPANDER
                # --------------------------
                with st.expander("✨ Enhancements", expanded=False):
                    st.subheader("🖌️ Adjustments")
                    brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.1)
                    contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.1)
                    saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
                    sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
                    
                    if st.button("✅ Apply Enhancements"):
                        enhancer_bright = ImageEnhance.Brightness(edited_img)
                        enhancer_contrast = ImageEnhance.Contrast(edited_img)
                        enhancer_color = ImageEnhance.Color(edited_img)
                        enhancer_sharp = ImageEnhance.Sharpness(edited_img)

                        edited_img = enhancer_bright.enhance(brightness)
                        edited_img = enhancer_contrast.enhance(contrast)
                        edited_img = enhancer_color.enhance(saturation)
                        edited_img = enhancer_sharp.enhance(sharpness)
                        
                        st.session_state["edited_image"] = edited_img
                        st.rerun()

                    # AI Enhancements
                    st.subheader("🤖 AI Enhancements")
                    col_ai = st.columns(2)
                    with col_ai[0]:
                        if st.button("🪄 Remove Background"):
                            with st.spinner("Removing background (AI)..."):
                                try:
                                    # input_bytes = pil_to_bytes(edited_img).getvalue()
                                    import io
                                    img_byte_arr = io.BytesIO()
                                    edited_img.save(img_byte_arr, format='PNG')
                                    input_bytes = img_byte_arr.getvalue()
                                    output_bytes = rembg.remove(input_bytes)
                                    edited_img = Image.open(io.BytesIO(output_bytes)).convert("RGB")
                                    st.session_state['imglab']["edited_image"] = edited_img
                                    st.success("✅ Background removed!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed: {e}")
                    with col_ai[1]:
                        if st.button("📈 Super Resolution (AI)"):
                            with st.spinner("Upscaling (AI)..."):
                                try:
                                    # Simple upscale (replace with Real-ESRGAN for pro AI upscaling)
                                    width = edited_img.width * 2
                                    height = edited_img.height * 2
                                    edited_img = edited_img.resize(
                                        (width, height), 
                                        Image.Resampling.LANCZOS
                                    )
                                    st.session_state['imglab']["edited_image"] = edited_img
                                    st.success("✅ Image upscaled 2x!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Failed: {e}")

                # --------------------------
                # CREATIVE EFFECTS EXPANDER
                # --------------------------
                with st.expander("🎨 Creative Effects", expanded=False):
                    selected_effect = st.selectbox(
                        "Select Effect",
                        [e.value for e in FilterEffect],
                        index=0
                    )
                    if st.button("✨ Apply Effect"):
                        effect = FilterEffect(selected_effect)
                        edited_img = apply_filter(edited_img, effect)
                        st.session_state['imglab']["edited_image"] = edited_img
                        st.rerun()
                # --------------------------
                # DOWNLOAD EXPANDER
                # --------------------------
                with st.expander("💾 Download", expanded=False):
                    st.subheader("Save Your Edit")
                    # ADD SVG TO FORMAT LIST
                    download_format = st.selectbox("Format", ["PNG", "JPG", "WEBP", "SVG"])
                    download_quality = st.slider("Quality (JPG/WEBP)", 50, 100, 95)
                    
                    # --------------------------
                    # HANDLE SVG (special case)
                    # --------------------------
                    if download_format == "SVG":
                        # For SVG: we export a simple SVG that embeds the image (works for any image)
                        def img_to_svg_bytes(img):
                            import io
                            import base64
                            buf = io.BytesIO()
                            img.save(buf, format="PNG")
                            b64 = base64.b64encode(buf.getvalue()).decode()
                            w, h = img.size
                            svg = f'''<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">
                                        <image href="data:image/png;base64,{b64}" width="{w}" height="{h}"/>
                                    </svg>'''
                            return io.BytesIO(svg.encode())

                        edited_bytes = img_to_svg_bytes(edited_img)
                        original_bytes = img_to_svg_bytes(original_img)
                    else:
                        # Normal image formats (PNG/JPG/WEBP)
                        edited_bytes = pil_to_bytes(edited_img, format=download_format)
                        original_bytes = pil_to_bytes(original_img, format=download_format)
                    
                    col_dl = st.columns(2)
                    with col_dl[0]:
                        st.download_button(
                            label="📥 Download Edited",
                            data=edited_bytes,
                            file_name=f"edited_image.{download_format.lower()}",
                            mime=f"image/{download_format.lower()}"
                        )
                    with col_dl[1]:
                        st.download_button(
                            label="📥 Download Original",
                            data=original_bytes,
                            file_name=f"original_image.{download_format.lower()}",
                            mime=f"image/{download_format.lower()}"
                        )

            # --------------------------
            # PREVIEW SECTION (MAIN)
            # --------------------------
            with col2:
                st.caption("Click & drag to select crop area (release to apply)")
                if enable_cropper:
                    edited_img = st_cropper(
                        edited_img,
                        realtime_update=True,    # SHOWS BOX WHILE DRAGGING
                        box_color="#1f77b4",     # BLUE BOX
                        aspect_ratio=aspect_ratio        # FREE CROP
                    )

                # Before/After comparison
                st.subheader("🔍 Before / After")
                col_comp = st.columns(2)
                with col_comp[0]:
                    st.image(original_img, caption="Original", width="stretch")
                with col_comp[1]:
                    st.image(edited_img, caption="Edited", width="stretch")
            # --------------------------
            # BATCH PROCESSING (BONUS)
            # --------------------------
            st.markdown("---")
            st.subheader("📦 Batch Processing (Pro)")
            st.caption("Apply the same edits to multiple images")
            batch_upload = st.file_uploader(
                "Upload multiple images",
                type=["jpg", "jpeg", "png", "webp"],
                accept_multiple_files=True,
                key="batch_uploader"
            )
            
            if batch_upload and st.button("🚀 Process Batch"):
                with st.spinner("Processing batch..."):
                    batch_zip = io.BytesIO()
                    from zipfile import ZipFile
                    
                    with ZipFile(batch_zip, "w") as zf:
                        for i, file in enumerate(batch_upload):
                            try:
                                img = Image.open(file).convert("RGB")
                                # Apply same edits as current image
                                # (resize, filters, enhancements)
                                img = img.resize((edited_img.width, edited_img.height), Image.Resampling.LANCZOS)
                                img = apply_filter(img, FilterEffect(selected_effect))
                                
                                # Save to zip
                                img_bytes = pil_to_bytes(img)
                                zf.writestr(f"batch_edited_{i+1}.png", img_bytes.getvalue())
                            except Exception as e:
                                st.warning(f"⚠️ Failed to process {file.name}: {e}")
                    
                    batch_zip.seek(0)
                    st.download_button(
                        label="📥 Download Batch (ZIP)",
                        data=batch_zip,
                        file_name="batch_edited_images.zip",
                        mime="application/zip"
                    )
                    st.success("✅ Batch processing complete!")

            # --------------------------
            # RESET ALL
            # --------------------------
            if st.button("🗑️ Reset All Edits", type="primary"):
                # st.session_state['imglab']["original_image"] = None
                st.session_state['imglab']["edited_image"] = original_img.copy()
                st.session_state['imglab']["crop_coords"] = (0, 0, 0, 0)
                st.session_state['imglab']['history'] = []
                st.success("✅ All edits reset!")
                st.rerun()
        
    # =========================================================
    # 🧾 HISTORY
    # =========================================================
    with hist_tab:
        if df is not None:
            with st.expander(f"Dataset History", expanded=False):
                for i, (change, df_state) in enumerate(st.session_state["history"][::-1]):
                    with st.expander(f"{i+1}. {change}", expanded=False):
                        st.dataframe(df_state.head(), width='stretch')
                        st.rerun()
        
        imglab_hist = st.session_state['imglab'].get('history', None)
        if imglab_hist is not None:
            with st.expander(f"Image Lab History", expanded=False):
                for i, (change, img_state) in enumerate(st.session_state['imglab']["history"][::-1]):
                    cols = st.columns(2)
                    with cols[0]: st.write(f"{i+1}. {change}");
                    with cols[1]: st.image(img_state, caption="Edited", width=300);
        if df is None and imglab_hist is None:
            st.warning("No Operations executed yet...")   


if __name__ == "__main__":
    render()
