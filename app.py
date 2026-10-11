from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = Path("model/best.pt")
EMOJI = {"apple": "🍎", "banana": "🍌", "mango": "🥭"}

st.set_page_config(page_title="Fruit Ripeness Classifier", page_icon="🍌")
st.title("🍌 Fruit Ripeness Classifier")
st.caption("Upload a photo of an apple, banana or mango. The model says which fruit it is and whether it is ripe.")

with st.sidebar:
    st.header("About this model")
    st.markdown(
        "Fine-tuned **YOLOv8n-cls** trained on 364 photos, 6 classes "
        "(apple / banana / mango × ripe / unripe)."
    )
    st.markdown(
        "| Model | Test accuracy |\n|---|---|\n"
        "| From scratch | 73.1% |\n"
        "| **Fine-tuned (this app)** | **95.2%** |"
    )
    st.caption("Test set: 145 held-out photos from the same source as the training photos.")
    st.markdown("**Limits**")
    st.markdown(
        "- One fruit per photo works best\n"
        "- Only 3 fruits, no 'overripe' class\n"
        "- Green apples and green mangoes can be confused\n"
        "- Photos from other sources may score lower"
    )

if not MODEL_PATH.exists():
    st.error("model/best.pt not found. Copy best.pt from Colab into a folder named 'model' next to app.py.")
    st.stop()


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

source = st.radio("Image source", ["Upload a photo", "Use camera"], horizontal=True)
if source == "Upload a photo":
    file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
else:
    file = st.camera_input("Take a photo")

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, width=400)

    with st.spinner("Looking at the fruit..."):
        res = model.predict(img, verbose=False)[0]

    probs = res.probs.data.cpu().numpy()
    names = res.names
    order = probs.argsort()[::-1]
    top = int(order[0])
    fruit, state = names[top].split("-")
    conf = float(probs[top])

    st.subheader(f"{EMOJI.get(fruit, '')} {fruit.capitalize()} - {state}")
    st.write(f"Confidence: **{conf:.0%}**")
    if conf < 0.6:
        st.warning("The model is not very sure about this one. Try a clearer photo with a single fruit.")

    df = pd.DataFrame({"confidence": [float(probs[i]) for i in order]},
                      index=[names[i] for i in order])
    st.bar_chart(df)
