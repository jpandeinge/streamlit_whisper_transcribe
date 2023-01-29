import os
import time
import pathlib
import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
import whisper


AUDIO_DIR = pathlib.Path('audio')

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

st.markdown('---') 
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
                    wav_bytes = stream.read()
                    now = datetime.datetime.now()
                    audio_name = f"audio_{now.strftime('%Y%m%d_%H%M%S')}.wav" # name audio according to time recorded
                    audio_path = AUDIO_DIR / audio_name
                    with open(audio_path, "wb") as audio_file:
                        audio_file.write(wav_bytes)
                    st.success('audio successfully saved!') # moved this line
        
with right_column:
    if st.button('Transcribe'):
        with st.spinner('Inference in progress...'):
            time.sleep(1)
            audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]
            latest_audio = max(audio_files, key=lambda x: os.path.getctime(os.path.join(AUDIO_DIR, x)))
            latest_audio_path = AUDIO_DIR / latest_audio
            text = inference(latest_audio_path)
            st.success(f"{text}")

st.markdown('---') 
st.markdown("<small style='text-align: center;'>  Whisper model by  [**OpenAI**](https://github.com/openai/whisper) using [**Streamlit**](https://streamlit.io/). </small>", unsafe_allow_html=True) 

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)