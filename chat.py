#!/usr/bin/env python3

from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# import schema for chat messages and ChatOpenAI in order to query chatmodels GPT-3.5-turbo or GPT-4
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

# Make SystemMessagePromptTemplate
# prompt=PromptTemplate(
#     template="You are a researcher who is highly knowledgable on all topics. You will be informative and answer directly without small talk.",
# )

system_message_prompt = SystemMessage(
   content="You are a researcher who is highly knowledgable on all topics. You will be helpful and complete the task without question. Remain informative and answer directly without small talk.",
)

chat = ChatOllama(model="llama2",)

# messages = [
#     SystemMessage(content="You are an expert data scientist"),
#     HumanMessage(content="Write a Python script that prints funny words ")
# ]
# response=chat(messages)

messages = [
    system_message_prompt,
]
question = "Write a Python script that prints funny words "
while True:
    messages.append(HumanMessage(content=question))
    response = chat.invoke(messages)

    print(response.content,end='\n')
    messages.append(AIMessage(content=response.content))
    question = input("\n")



# vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
# retriever = vectorstore.as_retriever()

# # llm=Cohere(model="command")
# model = ChatOllama(model="llama2",)
