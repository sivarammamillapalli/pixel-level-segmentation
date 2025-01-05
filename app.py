import json
import streamlit as st
from streamlit import session_state
import os
import numpy as np

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Input, Conv2D, MaxPooling2D, BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import warnings
from temp import model
warnings.filterwarnings('ignore')


session_state = st.session_state
if "user_index" not in st.session_state:
    st.session_state["user_index"] = 0


def signup(json_file_path="data.json"):
    st.title("Signup Page")
    with st.form("signup_form"):
        st.write("Fill in the details below to create an account:")
        name = st.text_input("Name:")
        email = st.text_input("Email:")
        age = st.number_input("Age:", min_value=0, max_value=120)
        sex = st.radio("Sex:", ("Male", "Female", "Other"))
        password = st.text_input("Password:", type="password")
        confirm_password = st.text_input("Confirm Password:", type="password")

        if st.form_submit_button("Signup"):
            if password == confirm_password:
                user = create_account(name, email, age, sex, password, json_file_path)
                session_state["logged_in"] = True
                session_state["user_info"] = user
            else:
                st.error("Passwords do not match. Please try again.")
def display_image(image, title='Image'):
    plt.figure(figsize=(6, 6))
    plt.title(title)
    
    # Check if the image is a TensorFlow tensor
    if isinstance(image, tf.Tensor):
        # Convert to NumPy array
        image = image.numpy()
        
    # Squeeze the image to remove single-dimensional entries from the shape
    image = np.squeeze(image)
    
    # Display the image
    plt.imshow(image, cmap='gray' if image.ndim == 2 else None)
    plt.axis('off')
    plt.show()
def check_login(username, password, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        for user in data["users"]:
            if user["email"] == username and user["password"] == password:
                session_state["logged_in"] = True
                session_state["user_info"] = user
                st.success("Login successful!")
                return user

        st.error("Invalid credentials. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error checking login: {e}")
        return None


def predict(uploaded_file):
    model = load_model('model.h5')
    bytes_data = uploaded_file.read()
    image = tf.io.decode_image(bytes_data, channels=3)
    image = tf.image.resize(image, (256, 256))
    image = tf.cast(image, tf.float32) / 255.0
    
    # Expand dimensions to simulate batch size of 1
    image = tf.expand_dims(image, axis=0)
    
    # Perform prediction
    pred_mask = model.predict(image)
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask


def initialize_database(json_file_path="data.json"):
    try:
        # Check if JSON file exists
        if not os.path.exists(json_file_path):
            # Create an empty JSON structure
            data = {"users": []}
            with open(json_file_path, "w") as json_file:
                json.dump(data, json_file)
    except Exception as e:
        print(f"Error initializing database: {e}")


def login(json_file_path="data.json"):
    st.title("Login Page")
    username = st.text_input("Username:")
    password = st.text_input("Password:", type="password")

    login_button = st.button("Login")

    if login_button:
        user = check_login(username, password, json_file_path)
        if user is not None:
            session_state["logged_in"] = True
            session_state["user_info"] = user
        else:
            st.error("Invalid credentials. Please try again.")

def get_user_info(email, json_file_path="data.json"):
    try:
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)
            for user in data["users"]:
                if user["email"] == email:
                    return user
        return None
    except Exception as e:
        st.error(f"Error getting user information: {e}")
        return None


def render_dashboard(user_info, json_file_path="data.json"):
    try:
        st.title(f"Welcome to the Dashboard, {user_info['name']}!")
        st.subheader("User Information:")
        st.write(f"Name: {user_info['name']}")
        st.write(f"Sex: {user_info['sex']}")
        st.write(f"Age: {user_info['age']}")
    except Exception as e:
        st.error(f"Error rendering dashboard: {e}")
def create_account(name, email, age, sex, password, json_file_path="data.json"):
    try:
        # Check if the JSON file exists or is empty
        if not os.path.exists(json_file_path) or os.stat(json_file_path).st_size == 0:
            data = {"users": []}
        else:
            with open(json_file_path, "r") as json_file:
                data = json.load(json_file)

        # Append new user data to the JSON structure
        user_info = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "password": password,

        }
        data["users"].append(user_info)

        # Save the updated data to JSON
        with open(json_file_path, "w") as json_file:
            json.dump(data, json_file, indent=4)

        st.success("Account created successfully! You can now login.")
        return user_info
    except json.JSONDecodeError as e:
        st.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        st.error(f"Error creating account: {e}")
        return None


def main(json_file_path="data.json"):
    st.sidebar.title("Real-time image segmentation for self-driving cars")
    page = st.sidebar.radio(
        "Go to",
        ("Signup/Login", "Dashboard", "Upload Image"),
        key="Real-time image segmentation for self-driving cars",
    )

    if page == "Signup/Login":
        st.title("Signup/Login Page")
        login_or_signup = st.radio(
            "Select an option", ("Login", "Signup"), key="login_signup"
        )
        if login_or_signup == "Login":
            login(json_file_path)
        else:
            signup(json_file_path)

    elif page == "Dashboard":
        if session_state.get("logged_in"):
            render_dashboard(session_state["user_info"])
        else:
            st.warning("Please login/signup to view the dashboard.")

    elif page == "Upload Image":
        if session_state.get("logged_in"):
            st.title("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose a image (PNG)", type=["png"]
            )
            if st.button("Upload") and uploaded_image is not None:
                st.image(uploaded_image, use_column_width=True)
                st.success("Image uploaded successfully!")
                image = predict(uploaded_image)[0]
                img = tf.keras.preprocessing.image.array_to_img(image.numpy())
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis('off')
                ax.set_title('Predicted Image')
                print(type(img))
                st.pyplot(fig)
                
if __name__ == "__main__":
    initialize_database()
    main()
