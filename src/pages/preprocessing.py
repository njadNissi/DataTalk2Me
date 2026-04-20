import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import src.core.utils as utils
import time

"""Use st.session_state['preprocessing']['key'] to store and retrieve related variables."""


def render():
    st.title("Data Preprocessing")

    # =====================================================
    # 🔐 STATE INIT
    # =====================================================
    if 'preprocessing' not in st.session_state: # No previously preprocessed data
        if "data" not in st.session_state:
            st.warning("Upload data first")
        else: # there is raw data
            st.session_state.setdefault('preprocessing', {})['data'] = st.session_state['data'].copy()
            utils.temp_show("✅ Your data has been loaded for preprocessing...", 'success', dur=0.5)
            st.rerun()
            
    else: # Found previously preprocessed data
        apply_col, reset_col = st.columns(2)
        with apply_col:
            if st.button("💾 Apply & Save all Preprocessing changes"):
                st.session_state['preprocessing']['APPLY_PREPROCESSING'] = True
                utils.temp_show("✅ Data preprocessing successful ✅", 'success', dur=0.5)
        # =====================================================
        # 🔄 RESET
        # =====================================================
        with reset_col:
            if st.button("♻️ Discard all preprocessing changes"):
                st.session_state.pop("preprocessing")
                utils.temp_show("🔄 Reset successful ✅", 'success', dur=0.5)
                st.rerun()

    df = st.session_state['preprocessing'].get("data")
    task_col, target_col = st.columns(2)
    with task_col:
        task = st.radio(
            label="Preprocess data for:",
            options=["classification", "regression"],
            horizontal=True, 
            index=0
        )
        # st.session_state["preprocessing"]["task"] = task
    
    # =====================================================
    # 🎯 TARGET
    # =====================================================
    # st.subheader("Target Selection")
    with target_col:
        selected_target = st.selectbox(
            "Select target column",
            df.columns,
            index=df.columns.get_loc(st.session_state.get("target", df.columns[-1]))
            if "target" in st.session_state else len(df.columns) - 1
        )
        # st.session_state["preprocessing"]["target"] = selected_target

    # =====================================================
    # 📊 NUMERIC FEATURES
    # =====================================================
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != selected_target]
    st.write("Encoded features:", [f for f in feature_cols if f in st.session_state["preprocessing"].get(f"{f}_enc", [])])

    # =====================================================
    # 🎯 FEATURE SCALING
    # =====================================================
    feat_sc1, feat_sc2 = st.columns([9, 1])
    with feat_sc1:
        selected_cols = st.multiselect(
            "Select features to SCALE",
            feature_cols,
            default=feature_cols
        )
    with feat_sc2:
        features_scaler_type = st.selectbox(
            "Feature Scaler",
            ["None", "StandardScaler", "MinMaxScaler"]
        )

    # =====================================================
    # 🎯 LABEL SCALING (NEW)
    # =====================================================
    sl1, sl2 = st.columns(2)
    with sl1:
        scale_label = st.checkbox(
            "Scale labels (for regression models)",
            disabled=task == "classification"
        )
    with sl2:
        label_scaler_type = st.selectbox(
            "Label Scaler",
            ["StandardScaler", "MinMaxScaler"],
            disabled=not scale_label
        )

    # =====================================================
    # ⚙️ PROCESSING
    # =====================================================
    df_processed = df.copy()

    features_scaler = None
    target_scaler = None

    X = df_processed[selected_cols]
    y = df_processed[selected_target]

    # -------- Train-test split (after encoding but before scaling to avoid data leakage) --------
    test_size = st.slider(
        f"Test size: {st.session_state['preprocessing'].get('test_size_val', 0.2)*100:.0f}% "\
            f"| Train size: {100 - st.session_state['preprocessing'].get('test_size_val', 0.2)*100:.0f}%",
        0.1, 0.5, 0.2,
        key="test_size_val"
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
     
    # -------- Feature scaling --------
    if features_scaler_type != "None" and len(selected_cols) > 0:
        features_scaler = StandardScaler() if features_scaler_type == "StandardScaler" else MinMaxScaler()
        X_train_scaled = features_scaler.fit_transform(X_train[selected_cols])
        X_test_scaled = features_scaler.transform(X_test[selected_cols])
        
        # Update df_processed with scaled values (align indices to avoid misalignment)
        df_processed.loc[X_train.index, selected_cols] = X_train_scaled
        df_processed.loc[X_test.index, selected_cols] = X_test_scaled

    # -------- Label scaling --------
    if scale_label and st.session_state['preprocessing']["task"] == "classification":
        target_scaler = StandardScaler() if label_scaler_type == "StandardScaler" else MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
        
        # Update df_processed with scaled target (align indices)
        df_processed.loc[y_train.index, selected_target] = y_train_scaled.ravel()
        df_processed.loc[y_test.index, selected_target] = y_test_scaled.ravel()


    # =====================================================
    # 📊 PREVIEW
    # =====================================================
    with st.expander("Preprocessed Data ovevrview", expanded=False):
        st.caption("Your full preprocessed data, ready for download 📥")
        st.dataframe(df_processed)
        st.caption("Preprocessed Data Description, ready for download 📥")
        st.dataframe(df_processed.describe())

    # =====================================================
    # 💾 APPLY
    # =====================================================
    if st.session_state['preprocessing'].get('APPLY_PREPROCESSING', False):
        st.session_state['preprocessing']['APPLY_PREPROCESSING'] = False

        st.session_state["preprocessing"]["task"] = task
        st.session_state["preprocessing"]["target"] = selected_target

        st.session_state["preprocessing"]["data"] = df_processed
        st.session_state["preprocessing"]["data_size"] = len(df_processed)
        st.session_state["preprocessing"]["test_size"] = test_size
        st.session_state["preprocessing"]["target"] = selected_target
        st.session_state["preprocessing"]["target_labels"] = sorted(set(y_train))
        st.session_state["preprocessing"]["feature_names"] = feature_cols

        # FIX 3: Store SCALED train/test splits (critical for modeling)
        st.session_state["preprocessing"]["X_train"] = df_processed.loc[X_train.index, selected_cols] # X_train_scaled if features_scaler_type != "None" else X_train
        st.session_state["preprocessing"]["X_test"] = df_processed.loc[X_test.index, selected_cols] # X_test_scaled if features_scaler_type != "None" else X_test
        st.session_state["preprocessing"]["y_train"] =  df_processed.loc[y_train.index, selected_target] # y_train_scaled.ravel() if scale_label else y_train
        st.session_state["preprocessing"]["y_test"] = df_processed.loc[y_test.index, selected_target] # y_test_scaled.ravel() if scale_label else y_test

        # ---- Feature scaling info ----
        if features_scaler is not None:
            st.session_state["preprocessing"]["features_scaler"] = features_scaler
            st.session_state["preprocessing"]["are_features_scaled"] = selected_cols
        else:
            st.session_state["preprocessing"].pop("features_scaler", None)
            st.session_state["preprocessing"].pop("are_features_scaled", None)

        # ---- Label scaling info ----
        if scale_label:
            st.session_state["preprocessing"]["target_scaler"] = target_scaler
            st.session_state["preprocessing"]["is_target_scaled"] = True
        else:
            st.session_state["preprocessing"].pop("target_scaler", None)
            st.session_state["preprocessing"]["is_target_scaled"] = False


        if "inference" in st.session_state: # reset if already visited inference.
            st.session_state['inference']["train_btn_clicked"] = False # In case it is true
            st.session_state['inference']["model"] = None # In case it is true

        utils.temp_show("✅ Changes applied", 'success', 1)

        
if __name__ == "__main__":
    render()