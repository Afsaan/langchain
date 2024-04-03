## integrate our code with openAI API
import os
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# streamlit framwork

st.title("Lanchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want")

# prompt template

first_input_prompt = PromptTemplate(
    input_variables = ["topic"]
)

## OPENAI LLMS
llm = OpenAI(temperature=0.8) #how unique the output will be (0- not unique and 1- unique data)

if input_text:
    st.write(llm(input_text))