import streamlit as st
import requests
import numpy as np
from IPython.core.display_functions import display
from gtts import gTTS
from IPython.display import Audio
from io import BytesIO
import tensorflow as tf
from PIL import Image
from model import get_caption_model, generate_caption, capture_image


@st.cache(allow_output_mutation=True)
def get_model():
    return get_caption_model()

caption_model = get_model()

img_url = capture_image()

if (img_url != "") or (img_url != None):
    img = Image.open(requests.get(img_url, stream=True).raw)
    st.image(img)

    img = np.array(img)
    pred_caption = generate_caption(img, caption_model)
    st.write(pred_caption)

    tts = gTTS(text=pred_caption, lang='en')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio = Audio(fp.read(), autoplay=True)
    display(audio)

