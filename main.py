from pydub import AudioSegment
#from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from elevenlabs import generate, play, voices 
from elevenlabs import set_api_key
from elevenlabslib import ElevenLabsUser
import streamlit as st
import uuid

from langchain import LLMChain, PromptTemplate, OpenAI
import os

# Load environment variables
#load_dotenv(find_dotenv())

# Set API Key For Gpt-3 and Hugging Face using st.secrets
OpenAIKey = st.secrets["OPENAI_API_KEY"]
HuggingFaceKey = st.secrets["HUGGINGFACE_API_KEY"]

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
    text = image_to_text(url)[0]["generated_text"]
    print(text)
    return text

def generated_story(scenario):
    template = """
    You are the famous comedian and podcaster Tim Dillon. You will generate a short story in the style of his podcast using a funny narrative with explicit language and an angry rant-like tone, the story should be no more than 100 words;
    Context: {scenario}
    Story:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    story_llm = LLMChain(llm=OpenAI(model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


def textToSpeech(story):
    # Error Handling: Print available user-generated voices
    try:
        available_voices = user.get_available_voices()
        print("Available Voices:", available_voices)
    except Exception as e:
        print("Error fetching available voices:", e)

    # Set API key for ElevenLabs
    set_api_key = st.secrets["ELEVENLABS_API_KEY"]
    user = ElevenLabsUser(set_api_key)
    voice = user.get_voice_by_ID("cgOzEASJmlEWHtXnZJ5q")

    # Generate the audio data
    result = voice.generate_audio_v2(story)

    # Assuming the audio data is the first element of the tuple
    audio_data = result[0]

    # Save the audio data to a file in the project folder
    random_id = str(uuid.uuid4())
    name = f"story_{random_id}.mp3"

    #Save the audio data to a file in the project folder
    with open(name, 'wb') as f:
        f.write(audio_data)
    return name

def main():
    st.set_page_config(page_title="Tim Dillon Image To Story", page_icon="📖", layout="wide")
    st.header("Tim Dillon Image To Story")
    uploaded_file = st.file_uploader("Upload an image...", type="jpg")
    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open (uploaded_file.name, 'wb') as f:
            f.write(bytes_data)
        st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generated_story(scenario)
        generated_file_name = textToSpeech(story)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)
        
        st.audio(generated_file_name)
        

if __name__ == "__main__":
    main()
