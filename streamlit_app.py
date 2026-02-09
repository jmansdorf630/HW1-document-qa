import streamlit as st
import importlib

st.set_page_config(page_title="HW Manager")
st.title("🔧 HW Manager")

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ("Home", "HW1", "HW2", "HW3"))

if page == "Home":
    st.header("Welcome to HW Manager")
    st.write("Use the sidebar to navigate to individual homework pages.")

elif page == "HW1":
    try:
        hw1 = importlib.import_module("HW.HW1")
        hw1.app()
    except Exception as e:
        st.error(f"Failed to load HW1: {e}")

elif page == "HW2":
    try:
        hw2 = importlib.import_module("HW.HW2")
        hw2.app()
    except Exception as e:
        st.error(f"Failed to load HW2: {e}")

elif page == "HW3":
    try:
        hw3 = importlib.import_module("HW.HW3")
        hw3.app()
    except Exception as e:
        st.error(f"Failed to load HW3: {e}")
