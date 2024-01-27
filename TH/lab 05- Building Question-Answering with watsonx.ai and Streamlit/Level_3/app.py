# Import environment loading library
from dotenv import load_dotenv
# Import IBMGen Library 
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from langchain.llms.base import LLM
# Import lang Chain Interface object
from langChainInterface import LangChainInterface
# Import langchain prompt templates
from langchain.prompts import PromptTemplate
# Import system libraries
import os
# Import streamlit for the UI 
import streamlit as st
import re

from translator import *

# Load environment vars
load_dotenv()

# Define credentials 
api_key = os.getenv("API_KEY", None)
ibm_cloud_url = os.getenv("IBM_CLOUD_URL", None)
project_id = os.getenv("PROJECT_ID", None)


if api_key is None or ibm_cloud_url is None or project_id is None:
    print("Ensure you copied the .env file that you created earlier into the same directory as this notebook")
else:
    creds = {
        "url": ibm_cloud_url,
        "apikey": api_key 
    }



# Define generation parameters 
params = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 300,
    GenParams.TEMPERATURE: 0.2,
    # GenParams.TOP_K: 100,
    # GenParams.TOP_P: 1,
    GenParams.REPETITION_PENALTY: 1
}

models = {
    "granite_chat":"ibm/granite-13b-chat-v1",
    "flanul": "google/flan-ul2",
    "llama2": "meta-llama/llama-2-70b-chat"
}
# define LangChainInterface model
llm = LangChainInterface(model=models["llama2"], credentials=creds, params=params, project_id=project_id)


# Title for the app
st.title('🤖 Our First Q&A Front End')
# Prompt box 
prompt = st.text_input('Enter your prompt here')
print(prompt)
# If a user hits enter
if prompt:
    # Pass the prompt to the llm
    text_to_model =  translate_to_thai(prompt, False)
    print('text_to_model')
    print(text_to_model)
    response_from_model = llm(text_to_model)
    print(response_from_model)

    # response_from_model = llm(prompt)
    text_to_user = translate_to_thai(response_from_model, True)
    # Write the output to the screen
    st.write(text_to_user)


# flanul
# ระยะเวลาการกู้ซื้อบ้าน นานเท่าไหร่?
# ข้อดี ของหนี้คืออะไร?
# ธนาคารโลกอยู่ที่ไหน ?
# สัดส่วนการออมที่เหมาะสมคือเท่าไหร่ ?

