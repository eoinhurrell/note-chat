#!/usr/bin/env python3

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_citation_fuzzy_match_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import ObsidianLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# messages = [
#     SystemMessage(content="You are a helpful assistant."),
#     HumanMessage(content="Hi AI, how are you today?"),
#     AIMessage(content="I'm great thank you. How can I help you?"),
#     HumanMessage(content="I'd like to understand string theory.")
# ]

# res = chat(messages)
# print(res.content)

# # add latest AI response to messages
# messages.append(res)

# # now create a new user prompt
# prompt = HumanMessage(
#     content="Why do physicists believe it can produce a 'unified theory'?"
# )
# # add to messages
# messages.append(prompt)

# # send to chat-gpt
# res = chat(messages)


vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
retriever = vectorstore.as_retriever()

template = """Answer the question based on your existing knowledge but with special attention to the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chat_model = ChatOllama(
    model="llama2",
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)
# print(chain.invoke("where did harrison work?"))

from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts.prompt import PromptTemplate

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

_inputs = RunnableParallel(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: get_buffer_string(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOllama(model="llama2",)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}
conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOllama(model="llama2",)

# print(conversational_qa_chain.invoke(
#     {
#         "question": "where did harrison work?",
#         "chat_history": [],
#     }
# )
#       )

print(conversational_qa_chain.invoke(
    {
        "question": "Citing specific ideas (using their path), explain long covid to me?",
        "chat_history": [
            HumanMessage(content="Do you understand covid?"),
            AIMessage(content="Yes, there are many idea documents on this topic."),
        ],
    }
))
