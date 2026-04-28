import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import src.core.utils as utils

def init_preprocessing_state():
    if 'preprocessing' not in st.session_state:
        st.session_state['preprocessing'] = {
            "data": None,
            "task": "classification",
            "target": None,
            "test_size": 0.2,
            "test_size_val": 0.2,
            "APPLY_PREPROCESSING": False,
            "features_scaler": None,
            "target_scaler": None,
            "are_features_scaled": [],
            "is_target_scaled": False,
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "feature_names": [],
            "target_labels": [],
            "data_size": 0,
            "features_scaler_type": "None",
            "label_scaler_type": "StandardScaler"
        }

def get_scaled_splits(X, y, test_size, feature_cols, features_scaler_type, scale_label, label_scaler_type, task):
    stratify = y if task == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X[feature_cols], y, test_size=test_size, random_state=42, stratify=stratify
    )

    features_scaler = None
    if features_scaler_type != "None" and len(feature_cols) > 0:
        features_scaler = StandardScaler() if features_scaler_type == "StandardScaler" else MinMaxScaler()
        X_train_scaled = features_scaler.fit_transform(X_train)
        X_test_scaled = features_scaler.transform(X_test)
        X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=feature_cols)
        X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=feature_cols)

    target_scaler = None
    if scale_label and task == "regression":
        target_scaler = StandardScaler() if label_scaler_type == "StandardScaler" else MinMaxScaler()
        y_train = target_scaler.fit_transform(y_train.values.reshape(-1,1)).ravel()
        y_test = target_scaler.transform(y_test.values.reshape(-1,1)).ravel()
        y_train = pd.Series(y_train, index=y_train.index, name=y.name)
        y_test = pd.Series(y_test, index=y_test.index, name=y.name)

    return X_train, X_test, y_train, y_test, features_scaler, target_scaler

def update_processed_dataframe(df_original, X_train, X_test, y_train, y_test, feature_cols, target_col):
    df_processed = df_original.copy()
    df_processed.loc[X_train.index, feature_cols] = X_train
    df_processed.loc[X_test.index, feature_cols] = X_test
    df_processed.loc[y_train.index, target_col] = y_train
    df_processed.loc[y_test.index, target_col] = y_test
    return df_processed

def render():
    st.title("Data Preprocessing")
    init_preprocessing_state()

    if st.session_state.get('data') is None:
        st.warning("⚠️ Upload data first!")
        return

    if st.session_state['preprocessing']['data'] is None:
        st.session_state['preprocessing']['data'] = st.session_state['data'].copy()
        st.session_state['preprocessing']['target'] = st.session_state['data'].columns[-1]
        utils.temp_show("✅ Data loaded", 'success', 0.5)

    df = st.session_state['preprocessing']['data']
    current_test_size = st.session_state['preprocessing']['test_size_val']
    current_target = st.session_state['preprocessing']['target']

    apply_col, reset_col = st.columns(2)
    with apply_col:
        if st.button("💾 Apply Preprocessing", type="primary"):
            st.session_state['preprocessing']['APPLY_PREPROCESSING'] = True
            utils.temp_show("✅ Applying...", 'success', 0.5)

    with reset_col:
        if st.button("♻️ Reset"):
            st.session_state['preprocessing'] = {
                "data": st.session_state['data'].copy(),
                "task": "classification",
                "target": df.columns[-1],
                "test_size": 0.2,
                "test_size_val": 0.2,
                "APPLY_PREPROCESSING": False,
                "features_scaler": None,
                "target_scaler": None,
                "are_features_scaled": [],
                "is_target_scaled": False,
                "X_train": None, "X_test": None,
                "y_train": None, "y_test": None,
                "feature_names": [], "target_labels": [],
                "data_size": 0,
                "features_scaler_type": "None",
                "label_scaler_type": "StandardScaler"
            }
            if "inference" in st.session_state:
                st.session_state["inference"]["model_name"] = ""
            utils.temp_show("🔄 Reset", 'success', 0.5)
            st.rerun()

    config_col1, config_col2 = st.columns(2)
    with config_col1:
        task = st.radio(
            "Task:", ["classification", "regression"], horizontal=True,
            index=0 if st.session_state['preprocessing']['task']=="classification" else 1
        )

    with config_col2:
        selected_target = st.selectbox("Target column", df.columns, index=df.columns.get_loc(current_target))

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != selected_target]

    sc1, sc2 = st.columns([9,1])
    with sc1:
        selected_feat_cols = st.multiselect("Features to scale", feature_cols, default=feature_cols)
    with sc2:
        fst = st.selectbox("Scaler", ["None", "StandardScaler", "MinMaxScaler"], index=["None","StandardScaler","MinMaxScaler"].index(st.session_state['preprocessing']['features_scaler_type']))

    ls1, ls2 = st.columns(2)
    with ls1:
        scale_label = st.checkbox("Scale target (regression only)", value=st.session_state['preprocessing']['is_target_scaled'], disabled=task=="classification")
    with ls2:
        lst = st.selectbox("Label scaler", ["StandardScaler","MinMaxScaler"], disabled=not scale_label)

    test_size = st.slider("Test size", 0.1,0.5, current_test_size, format="%.1f")

    X = df[feature_cols]
    y = df[selected_target]

    try:
        X_train, X_test, y_train, y_test, fs, ts = get_scaled_splits(
            X,y,test_size,selected_feat_cols,fst,scale_label,lst,task
        )
        dfp = update_processed_dataframe(df,X_train,X_test,y_train,y_test,selected_feat_cols,selected_target)
    except ValueError as ve:
        if "has only 1 member, which is too few" in str(ve):
            st.error(f"⚠️ ValueError: The least populated class in `{selected_target}` has only 1 member, which is too few. The minimum number of groups for any class cannot be less than 2.")
            return

    with st.expander("Preview"):
        st.dataframe(dfp)

    if st.session_state['preprocessing']['APPLY_PREPROCESSING']:
        st.session_state['preprocessing']['APPLY_PREPROCESSING'] = False
        st.session_state['preprocessing'].update({
            "task": task, "target": selected_target, "data": dfp,
            "data_size": len(dfp), "test_size": test_size, "test_size_val": test_size,
            "target_labels": sorted(set(y_train)) if task=="classification" else [],
            "feature_names": selected_feat_cols,
            "are_features_scaled": selected_feat_cols if fst!="None" else [],
            "is_target_scaled": scale_label and task=="regression",
            "features_scaler_type": fst, "label_scaler_type": lst,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "features_scaler": fs, "target_scaler": ts
        })
        if "inference" in st.session_state:
            st.session_state["inference"]["model_name"] = ""
        utils.temp_show("✅ Ready for inference", 'success', 2)
        st.rerun()

if __name__ == "__main__":
    render()