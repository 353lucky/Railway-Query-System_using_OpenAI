import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from st_audiorec import st_audiorec
import whisper
from transformers import VitsModel, AutoTokenizer
import torch
import scipy
import speech_recognition as sr





load_dotenv()

# Function to initialize SQL database and OpenAI
def initialize():
    # username = "user"
    # password = "password"
    # host = "localhost"
    # port = "5432"
    # mydatabase = "train"

    # pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{mydatabase}"
    # db = SQLDatabase.from_uri(pg_uri)
    db = SQLDatabase.from_uri("sqlite:///train_information.db")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

    return agent_executor

# Speech to Text
def save_audio(wav_audio_data,audio_file_path):
    
    with open(audio_file_path, "wb") as f:
        f.write(wav_audio_data)

    # data = whisper.load_audio(wav_audio_data)
    # speech_to_text_model = whisper.load_model("medium")
    # transcribed_text = speech_to_text_model.transcribe(data)

    # return transcribed_text["text"]

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return text


# Text to Speech
def generate_speech(text):
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        output = model(**inputs).waveform
    
    scipy.io.wavfile.write("speech.wav", rate=model.config.sampling_rate, data=output.float().numpy().T)

# Runs Streamlit
def main():
    agent_executor = initialize()

    st.set_page_config(page_title="Railway Query System", layout="centered")

    st.title("Railway Query System")

    st.header("Ask any question:")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        audio_file_path = "audio.wav"
        save_audio(wav_audio_data,audio_file_path)
        transcribed_text = transcribe_audio(audio_file_path)
        st.write("Transcribed Text:")
        st.write(transcribed_text)
        
        input_to_llm = transcribed_text
        # input_to_llm = "Where is Goa Express right now?"
        result = agent_executor.invoke({"input": input_to_llm})
        
        st.header("AI Response")
        st.write(result["output"])
        
        generate_speech(result["output"])

        st.audio("speech.wav", format="audio/wav")

if __name__ == "__main__":
    main()
