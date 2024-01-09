#!/usr/bin/env python3

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI

from langchain.memory import ConversationBufferMemory
import pickle


from langchain.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from rich.console import Console
from rich.prompt import Prompt

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the most recent state of the union address.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """You are an AI assistant for answering questions about the most recent state of the union address.
You are given the following extracted parts of a long document and a question. Provide a conversational answer.
If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not about the most recent state of the union, politely inform them that you are tuned to only answer questions about the most recent state of the union.
Lastly, answer the question as if you were a pirate from the south seas and are just coming back from a pirate expedition where you found a treasure chest full of gold doubloons.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])


def load_retriever():
    loader = TextLoader("./data/20211207233831-cormac_mccarthy_embodied_in_a_vicious_world.md")
    docs = loader.load()

    embeddings=OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents, embeddings)
    # vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
    retriever = vectorstore.as_retriever()
    # with open("vectorstore.pkl", "rb") as f:
    #     vectorstore = pickle.load(f)
    # retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def get_basic_qa_chain():
    # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = ChatOllama(model="llama2",)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory)
    return model


def get_custom_prompt_qa_chain():
    # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = ChatOllama(model="llama2",)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/6635
    # see: https://github.com/langchain-ai/langchain/issues/1497
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_condense_prompt_qa_chain():
    # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = ChatOllama(model="llama2",)
    retriever = load_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    # see: https://github.com/langchain-ai/langchain/issues/5890
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model


def get_qa_with_sources_chain():
    # llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = ChatOllama(model="llama2",)
    retriever = load_retriever()
    history = []
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True)

    def model_func(question):
        # bug: this doesn't work with the built-in memory
        # hacking around it for the tutorial
        # see: https://github.com/langchain-ai/langchain/issues/5630
        new_input = {"question": question['question'], "chat_history": history}
        print(new_input)
        result = model.invoke(new_input)
        history.append((question['question'], result['answer']))
        return result

    return model_func


chain_options = {
    "basic": get_basic_qa_chain,
    "with_sources": get_qa_with_sources_chain,
    "custom_prompt": get_custom_prompt_qa_chain,
    "condense_prompt": get_condense_prompt_qa_chain
}


if __name__ == "__main__":
    c = Console()
    model = Prompt.ask("Which QA model would you like to work with?",
                       choices=list(chain_options.keys()),
                       default="with_sources")
    chain = chain_options[model]()

    c.print("[bold]Chat with your docs!")
    c.print("[bold red]---------------")

    while True:
        default_question = "what?"
        question = Prompt.ask("Your Question: ", default=default_question)
        # change this line if you're using RetrievalQA
        # input = query
        # output = result
        result = chain({"question": question})
        c.print("[green]Answer: [/green]" + result['answer'])

        # include a bit more if we're using `with_sources`
        if model == "with_sources" and result.get('source_documents', None):
            c.print("[green]Sources: [/green]")
            for doc in result['source_documents']:
                c.print(f"[bold underline green]{doc.metadata['source']}")
                c.print("[green]" + doc.page_content.replace('[','<'))
        c.print("[bold red]---------------")
