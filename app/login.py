import streamlit as st
import hashlib
import pandas as pd

def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# Page Configuration
st.set_page_config(page_title="Project FORESIGHT Login", page_icon=" ", layout="centered", initial_sidebar_state="collapsed")


# Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "role" not in st.session_state:
    st.session_state.role = "Guest"


# Password Hashing Function
def hash_password(password):
    """
    Convert plain password into SHA256 hash.
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# Demo User Database
# Replace this with database in production
users = pd.DataFrame(
    [
        {"username": "admin", "password": hash_password("admin123"), "role": "Administrator"},
        {"username": "manager", "password": hash_password("manager123"), "role": "Manager"},
        {"username": "user", "password": hash_password("user123"), "role": "User"}
    ]
)


# Header
st.title("Project FORESIGHT")
st.subheader("Demand Forecasting & Inventory Intelligence")
st.markdown("---")
st.info("""Please login to access the dashboard.""")


# Login Form
with st.form("login_form"):
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    remember_me = st.checkbox("Remember Me")
    login_button = st.form_submit_button("Login")
st.markdown("---")

st.caption("Project FORESIGHT | Secure Login")


# Authentication Logic
if login_button:
    if username.strip() == "" or password.strip() == "":
        st.error("Please enter Username and Password.")

    else:
        hashed_password = hash_password(password)
        matched_user = users[(users["username"] == username) & (users["password"] == hashed_password)]

        if len(matched_user) > 0:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = matched_user.iloc[0]["role"]

            if remember_me:
                st.session_state.remember_me = True
            st.success(f"Welcome {username}")

        else:
            st.error("Invalid Username or Password.")


# Session Login
if st.session_state.logged_in:
    st.markdown("---")
    st.success(f"""
        Logged in Successfully 
        User : {st.session_state.username}
        Role : {st.session_state.role}
    """)


# Dashboard Redirect
    st.info("Click below to open Dashboard.")
    if st.button("Open Dashboard"):
        try:
            st.switch_page("pages/1_Dashboard.py")

        except Exception:
            st.warning("""
                Unable to redirect automatically.
                Open Dashboard manually
                from Streamlit sidebar.
            """)


# Logout
    st.markdown("---")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = "Guest"

        if "remember_me" in st.session_state:
            del st.session_state["remember_me"]

        st.success("Logged out successfully.")
        st.rerun()


# Error Handling
try:
    pass
except Exception as e:
    st.error(f"Unexpected Error : {e}")


# Remember Me
if st.session_state.get("remember_me", False):
    st.sidebar.success("Remember Me Enabled")

else:
    st.sidebar.info("Remember Me Disabled")


# User Profile
if st.session_state.logged_in:
    st.markdown("---")
    st.header("User Profile")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Username", st.session_state.username)

    with col2:
        st.metric("Role", st.session_state.role)

    st.markdown("---")


# Role Based Access
if st.session_state.logged_in:
    role = st.session_state.role
    st.header("Access Permissions")

    if role == "Administrator":
        st.success("""
            Administrator Access
            1. Dashboard
            2. Forecast
            3. Risk Scoring
            4. Reports
            5. User Management
            6. Settings
            7. Download Reports
        """)

    elif role == "Manager":
        st.info("""
            Manager Access
            1. Dashboard
            2. Forecast
            3. Risk Scoring
            4. Reports
            5 Download CSV
        """)

    else:
        st.warning("""
            User Access
            1. Dashboard
            2. Forecast
            3. View Reports
        """)

st.markdown("---")


# Navigation
if st.session_state.logged_in:
    st.header("Quick Navigation")
    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Dashboard"):
            try:
                st.switch_page("pages/1_Dashboard.py")
            except:
                st.info("Open Dashboard from sidebar.")

    with c2:
        if st.button("Forecast"):
            try:
                st.switch_page("pages/2_Forecast.py")
            except:
                st.info("Open Forecast from sidebar.")

    with c3:
        if st.button("Risk Scoring"):
            try:
                st.switch_page("pages/3_Risk_Scoring.py")
            except:
                st.info("Open Risk Scoring from sidebar.")

st.markdown("---")


# Professional UI
st.header("System Information")

system = {
    "Project": "Project FORESIGHT",
    "Version": "1.0",
    "Framework": "Streamlit",
    "Machine Learning": "XGBoost + LightGBM",
    "Forecasting": "Demand Forecasting",
    "Risk Engine": "Inventory Intelligence"
}

st.table(system)
st.markdown("---")


# Footer
st.caption("Project FORESIGHT | Secure Authentication System | Version 1.0")