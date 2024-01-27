import re
import os
import json
from dotenv import load_dotenv
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams


def question_prompt(sentence):
    prompt = f'''
<s>[INST] <<SYS>>
INSTRUCTION:
คุณเป็นผู้เชี่ยวชาญด้านการเงิน และตอบเป็นภาษาไทยในรูปแบบสั้น ๆ
Please answer in Thai language you are the financial advisor. Answer in short and brief.
Produce the answer using the steps as below.
        Step 1: Understand the QUESTION.
        Step 2: WRITE the brief and easy ENGLISH answer.
        Step 3: Translate the brief and easy ENGLISH answer into Thai language.
        Step 4: Rewrite Step 4 into Thai simple Answer for kid in Thai language.
AVOID the new line as much as possible.
Start your response with 'Sure, I can answer in Thai. Here's my response:'
<</SYS>>
INPUT:
QUESTION: {sentence}
Step 1: Understand QUESTION: 
Step 2: ENGLISH BRIEF ANSWER:
Step 3: THAI TRANSLATED ANSWER:
Step 5: REWRITE INTO SIMPLE THAI ANSWER: 
[/INST]
    '''
    return prompt


def get_llm_model(creds, project_id):
    decoding_method="greedy"
    max_new_tokens=500
    min_new_tokens=1
    temperature=1.0
    repetition_penalty=1.0

    model_params = {
        GenParams.DECODING_METHOD: decoding_method,
        GenParams.MIN_NEW_TOKENS: min_new_tokens,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        GenParams.RANDOM_SEED: 42,
        GenParams.TEMPERATURE: temperature,
        GenParams.REPETITION_PENALTY: repetition_penalty,
    }
    model = Model(
        model_id="meta-llama/llama-2-70b-chat",
        params=model_params,
        credentials=creds,
        project_id=project_id)

    return model