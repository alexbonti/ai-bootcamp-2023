import logging
import os
import pickle
import tempfile
import textwrap

import streamlit as st
from dotenv import load_dotenv
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import (HuggingFaceHubEmbeddings,
                                  HuggingFaceInstructEmbeddings)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from PIL import Image
from googletrans import Translator
import requests

from langChainInterface import LangChainInterface

# Most GENAI logs are at Debug level.
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

st.set_page_config(
    page_title="Retrieval Augmented Generation",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.header("Retrieval Augmented Generation with watsonx.ai 💬")
# chunk_size=1500
# chunk_overlap = 200

load_dotenv()

handler = StdOutCallbackHandler()

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

GEN_API_KEY = os.getenv("GENAI_KEY", None)

# Sidebar contents
with st.sidebar:
    st.title("RAG App")
    st.markdown('''
    ## About
    This app is an LLM-powered RAG built using:
    - [IBM Generative AI SDK](https://github.com/IBM/ibm-generative-ai/)
    - [HuggingFace](https://huggingface.co/)
    - [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) LLM model
 
    ''')
    st.write('Powered by [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai)')
    image = Image.open('watsonxai.jpg')
    st.image(image, caption='Powered by watsonx.ai')
    max_new_tokens= st.number_input('max_new_tokens',1,1024,value=300)
    min_new_tokens= st.number_input('min_new_tokens',0,value=15)
    repetition_penalty = st.number_input('repetition_penalty',1,2,value=2)
    decoding = st.text_input(
            "Decoding",
            "greedy",
            key="placeholder",
        )
    
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)

def translate_large_text(text, translate_function, choice, max_length=500):
    """
    Break down large text, translate each part, and merge the results.

    :param text: str, The large body of text to translate.
    :param translate_function: function, The translation function to use.
    :param max_length: int, The maximum character length each split of text should have.
    :return: str, The translated text.
    """
    
    # Split the text into parts of maximum allowed character length.
    text_parts = textwrap.wrap(text, max_length, break_long_words=True, replace_whitespace=False)

    translated_text_parts = []

    for part in text_parts:
        # Translate each part of the text.
        translated_part = translate_function(part, choice)  # Assuming 'False' is a necessary argument in the actual function.
        translated_text_parts.append(translated_part)

    # Combine the translated parts.
    full_translated_text = ' '.join(translated_text_parts)

    return full_translated_text


def translate_to_thai(sentence: str, choice: bool) -> str:
    """
    Translate the text between English and Thai based on the 'choice' flag.
    
    Args:
        sentence (str): The text to translate.
        choice (bool): If True, translates text to Thai. If False, translates to English.

    Returns:
        str: The translated text.
    """
    translator = Translator()
    try:
        if choice:
            # Translate to Thai
            translate = translator.translate(sentence, dest='th')
        else:
            # Translate to English
            translate = translator.translate(sentence, dest='en')
        return translate.text
    except Exception as e:
        # Handle translation-related issues (e.g., network error, unexpected API response)
        raise ValueError(f"Translation failed: {str(e)}") from e

@st.cache_data
def read_pdf(uploaded_files, chunk_size=250, chunk_overlap=20):
    translated_docs = []

    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as temp_file:
            # Write content to the temporary file
            temp_file.write(bytes_data)
            filepath = temp_file.name

            with st.spinner('Waiting for the file to upload'):
                loader = PyPDFLoader(filepath)
                data = loader.load()

                for doc in data:
                    # Extract the content of the document
                    content = doc.page_content

                    # Translate the content
                    translated_content = translate_large_text(content, translate_to_thai, False)

                    # Replace original content with translated content
                    doc.page_content = translated_content
                    translated_docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(translated_docs)
    
    return docs


@st.cache_data
def read_push_embeddings():
    embeddings = HuggingFaceHubEmbeddings(repo_id="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("db.pickle"):
        with open("db.pickle",'rb') as file_name:
            db = pickle.load(file_name)
    else:     
        db = FAISS.from_documents(docs, embeddings)
        with open('db.pickle','wb') as file_name  :
             pickle.dump(db,file_name)
        st.write("\n")
    return db

# show user input
if user_question := st.text_input(
    "Ask a question about your Policy Document:"
):  
    translated_user_input = translate_to_thai(user_question, False)
    docs = read_pdf(uploaded_files)
    db = read_push_embeddings()
    docs = db.similarity_search(translated_user_input)
    params = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 30,
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.TEMPERATURE: 0.0,
        # GenParams.TOP_K: 100,
        # GenParams.TOP_P: 1,
        GenParams.REPETITION_PENALTY: 1
    }
    print('docs'+"*"*5)
    print(docs)
    print("*"*5)
    model_llm = LangChainInterface(model=ModelTypes.LLAMA_2_70B_CHAT.value, credentials=creds, params=params, project_id=project_id)
    chain = load_qa_chain(model_llm, chain_type="stuff")

    response = chain.run(input_documents=docs, question=translated_user_input)
    translated_response = translate_to_thai(response, True)

    st.text_area(label="Model Response", value=translated_response, height=100)
    st.write()

