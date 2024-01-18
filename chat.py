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
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

# import schema for chat messages and ChatOpenAI in order to query chatmodels GPT-3.5-turbo or GPT-4
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Make SystemMessagePromptTemplate
# prompt=PromptTemplate(
#     template="You are a researcher who is highly knowledgable on all topics. You will be informative and answer directly without small talk.",
# )


vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
retriever = vectorstore.as_retriever()
chat = ChatOllama(
    model="mistral",
)


# messages = [
#     SystemMessage(content="You are an expert data scientist"),
#     HumanMessage(content="Write a Python script that prints funny words ")
# ]
# response=chat(messages)
def get_qa_with_sources_chain():
    # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = ChatOllama(
        model="mistral",
    )
    vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
    retriever = vectorstore.as_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, return_source_documents=True
    )

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question["question"], "chat_history": history}
        print(new_input)
        result = model.invoke(new_input)
        history.append((question["question"], result["answer"]))
        return result

    return model_func


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
