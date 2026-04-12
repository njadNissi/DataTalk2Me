import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


def render():
    st.title("Data Preprocessing")

    # =====================================================
    # 🔐 STATE INIT
    # =====================================================
    if "raw_data" not in st.session_state:
        if "data" in st.session_state:
            st.session_state["raw_data"] = st.session_state["data"].copy()
        else:
            st.warning("Upload data first")
            return

    raw_df = st.session_state["raw_data"]
    df = raw_df.copy()

    task_col, target_col = st.columns(2)
    with task_col:
        task = st.radio(
            label="Preprocess data for:",
            options=["classification", "regression"],
            horizontal=True, 
            index=0
        )
        st.session_state["task"] = task
    
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
        st.session_state["target"] = selected_target

    # =====================================================
    # 📊 NUMERIC FEATURES
    # =====================================================
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != selected_target]
    st.write("Encoded features:", [f for f in feature_cols if f in st.session_state.get(f"{f}_enc", [])])

    # =====================================================
    # 🎯 FEATURE SCALING
    # =====================================================
    st.subheader("Feature Scaling")

    selected_cols = st.multiselect(
        "Select features to SCALE",
        feature_cols,
        default=feature_cols
    )
    features_scaler_type = st.selectbox(
        "Feature Scaler",
        ["None", "StandardScaler", "MinMaxScaler"]
    )

    # =====================================================
    # 🎯 LABEL SCALING (NEW)
    # =====================================================
    st.subheader("Label Scaling")
    sl1, sl2 = st.columns(2)

    with sl1:
        scale_label = st.checkbox(
            "Scale label (for regression models)",
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
        f"Test size: {st.session_state.get('test_size_val', 0.2)*100:.0f}% | Train size: {100 - st.session_state.get('test_size_val', 0.2)*100:.0f}%",
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
    if scale_label and st.session_state["task"] == "classification":
        target_scaler = StandardScaler() if label_scaler_type == "StandardScaler" else MinMaxScaler()
        y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))
        
        # Update df_processed with scaled target (align indices)
        df_processed.loc[y_train.index, selected_target] = y_train_scaled.ravel()
        df_processed.loc[y_test.index, selected_target] = y_test_scaled.ravel()


    # =====================================================
    # 📊 PREVIEW
    # =====================================================
    st.write("Preprocessed Data Preview")
    st.dataframe(df_processed.head())
    st.write("Preprocessed Data Description")
    st.dataframe(df_processed.describe())

    # =====================================================
    # 💾 APPLY
    # =====================================================
    apply_col,_, _, reset_col = st.columns(4)
    with apply_col:
        if st.button("💾 Apply Preprocessing"):

            st.session_state["data"] = df_processed
            st.session_state["target"] = selected_target
            st.session_state["feature_names"] = feature_cols
            st.session_state["test_size"] = test_size

            # FIX 3: Store SCALED train/test splits (critical for modeling)
            st.session_state["X_train"] = df_processed.loc[X_train.index, selected_cols] # X_train_scaled if features_scaler_type != "None" else X_train
            st.session_state["X_test"] = df_processed.loc[X_test.index, selected_cols] # X_test_scaled if features_scaler_type != "None" else X_test
            st.session_state["y_train"] =  df_processed.loc[y_train.index, selected_target] # y_train_scaled.ravel() if scale_label else y_train
            st.session_state["y_test"] = df_processed.loc[y_test.index, selected_target] # y_test_scaled.ravel() if scale_label else y_test

            # ---- Feature scaling info ----
            if features_scaler is not None:
                st.session_state["features_scaler"] = features_scaler
                st.session_state["are_features_scaled"] = selected_cols
            else:
                st.session_state.pop("features_scaler", None)
                st.session_state.pop("are_features_scaled", None)

            # ---- Label scaling info ----
            if scale_label:
                st.session_state["target_scaler"] = target_scaler
                st.session_state["is_target_scaled"] = True
            else:
                st.session_state.pop("target_scaler", None)
                st.session_state["is_target_scaled"] = False

            st.success("✅ Changes applied")
    with reset_col:
        # =====================================================
        # 🔄 RESET
        # =====================================================
        if st.button("♻️ Reset to Original"):

            st.session_state["data"] = st.session_state["raw_data"].copy()

            # Clear all pipeline info
            for key in ["features_scaler", "are_features_scaled", "target_scaler", "is_target_scaled"]:
                st.session_state.pop(key, None)

            st.success("🔄 Reset successful")
            st.rerun()

        
if __name__ == "__main__":
    render()