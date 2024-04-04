## integrate our code with openAI API
import os
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

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
    template = """
                when
                {person} 
                was born . i want to output has date month year only! no intrested in any other output"""
)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key="dob")

third_input_prompt = PromptTemplate(
    input_variables = ["dob"],
    template = """tell me any 5 major events that happend around {dob}
                and it should be in points"""
)

chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key="event")

# combining both the chain and set a seq
parent_chain = SequentialChain(chains=[chain, chain2, chain3], input_variables=["name"], output_variables=["person", "dob", "event"], verbose=True)

#memeory
person_memeory = ConversationBufferMemory(input_key="person", memory_key='chat_history')
dob_memeory = ConversationBufferMemory(input_key="person", memory_key='chat_history')


# we need to give the value in key value pair
if input_text:
    st.write(parent_chain({"name" : input_text}))