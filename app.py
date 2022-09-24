

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import numpy as np
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import whisper


# set the page layout
st.set_page_config(
    page_title="Whisper",
    page_icon="🤫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center;'> 🤫 Whisper</h1>", unsafe_allow_html=True)
st.markdown("<small style='text-align: center;'> Whisper is a general-purpose audio speech recognition model. Its trained on a large dataset of diverse audio and is also a  mili-task model that can perform multilingual speech recognition as well as speech translation and language identification. It is also a  mili-task model that can perform multilingual speech recognition as well as speech translation and language identification</small>", unsafe_allow_html=True)



model = whisper.load_model("small")

def inference(audio):
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    _, probs = model.detect_language(mel)
    
    options = whisper.DecodingOptions(fp16 = False)
    result = whisper.decode(model, mel, options)
    
    print(result.text)
    return result.text


parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
st_audiorec = components.declare_component("st_audiorec", path=build_dir)


left_column, right_column  = st.columns(2)
with left_column:
    # STREAMLIT AUDIO RECORDER Instance
    val = st_audiorec()

    if isinstance(val, dict):  # retrieve audio data
        with st.spinner('retrieving audio-recording...'):
            ind, val = zip(*val['arr'].items())
            ind = np.array(ind, dtype=int)  # convert to np array
            val = np.array(val)             # convert to np array
            sorted_ints = val[ind]
            stream = BytesIO(b"".join([int(v).to_bytes(1, "big") for v in sorted_ints]))
            # read the stream as a wav file
            audio = stream.read()

with right_column:
    if st.button('Transcribe'):
        with st.spinner('Transcribing...'):
            text = inference(audio)
            st.write(text)
