import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import subprocess
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

path_to_snn_latent_script = os.path.abspath(os.path.join(current_dir, "../mnist/create_latent_vectors.py"))
user_images_dir = os.path.abspath(os.path.join(current_dir, "../user_images/"))
image_path = os.path.join(user_images_dir, "digit.png")

os.makedirs(user_images_dir, exist_ok=True)


st.title('My Digit Recognizer')

SIZE = 192
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

selected_model = st.selectbox(
    "Válassz modelt!",
    ("Spiking Neural Network (Mushroom Body)", "Echo State Network (Watts-Strogatz)", "Echo State Network (Barabási-Albert)", "Echo State Network (Erdős-Rényi)"),
)

st.write("Kiválasztott modell:", selected_model)

num_of_results = st.selectbox(
    "Válassz, hogy hány találatot szeretnél megjeleníteni!",
    (3, 5, 7),
)

st.write("Max. találatok száma:", num_of_results)

if st.button('START'):
    if selected_model == "Spiking Neural Network (Mushroom Body)":
        cv2.imwrite(image_path, img)
        script_folder = os.path.dirname(path_to_snn_latent_script)
        
        try:
            subprocess.run(
                ["python3", path_to_snn_latent_script, "--image_path", image_path],
                cwd=script_folder,
                check=True
            )
            st.success("Latent vektor legenerálva!")
        except subprocess.CalledProcessError as e:
            st.error(f"Error a latent vektor generálásban: {e}")