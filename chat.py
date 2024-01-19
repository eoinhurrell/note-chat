#!/usr/bin/env python3

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
retriever = vectorstore.as_retriever()
chat = ChatOllama(
    model="mistral",
)


template = """Use the following search results from my notes to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | chat
    | StrOutputParser()
)
question = "Write a Python script that prints funny words "
while True:
    # response = chat.invoke(messages)
    response = rag_chain.invoke(question)

    print(response, end="\n")
    question = input("\n")
