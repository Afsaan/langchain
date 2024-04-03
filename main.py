## integrate our code with openAI API
import os
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SimpleSequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# streamlit framwork

st.title("Lanchain Demo with OpenAI API")
input_text = st.text_input("Search the topic you want")

# prompt template

first_input_prompt = PromptTemplate(
    input_variables = ["name"],
    template = "Tell me about celebrity {name}"
)

## OPENAI LLMS
llm = OpenAI(temperature=0.8) #how unique the output will be (0- not unique and 1- unique data)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key="person")

second_input_prompt = PromptTemplate(
    input_variables = ["person"],
    template = "when {person} was born"
)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob")

# combining both the chain and set a seq
parent_chain = SimpleSequentialChain(chains=[chain, chain2], verbose=True)

if input_text:
    st.write(parent_chain.run(input_text))