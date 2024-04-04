import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS #vector databases
from constant import openai_key

# load the key in the env
os.environ["OPENAI_API_KEY"] = openai_key

# load the pdf
pdfreader = PdfReader('data/budget_speech.pdf')

from typing_extensions import Concatenate

# read text from the pdf

raw_text = ""

for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# print(raw_text)

# we nedd to split the text using character text split such that it does not increase the token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800, # size of a sentence
    chunk_overlap = 200, # 200 overlap of chiunk from next sentence at last
    length_function = len,
)

texts = text_splitter.split_text(raw_text)

print(len(texts))

# download embedding from OPENAI
embeddings = OpenAIEmbeddings()

document_search = FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain =load_qa_chain(OpenAI(), chain_type='stuff')

query = "what is the vision for amrit kaal"
docs = document_search.similarity_search(query)

print(chain.run(input_documents = docs, question=query))