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
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.embeddings.cohere import CohereEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

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


#create custom prompt for your use case
prompt_template="""You are Delores, an AI customer service agent that talks to people querying the Tasks note library.
Answer the questions using the facts provided. You always uses pleasantries and polite greetings like "Good morning"
and "Good afternoon". You will receive a $100 tip for good performance. Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(prompt_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

chain_type_kwargs = {"prompt": prompt}

# llm=Cohere(model="command")
llm = ChatOllama(
    model="llama2",
)

#build your chain for RAG+C
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # retriever=vector_store.as_retriever(),
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

#print your results with Markup language
def print_result(result):
  output_text = f"""### Question:
  ### Answer:
  {result['answer']}
  ### Sources:
  {result['sources']}
  ### All relevant sources:
  {'\n'.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
  """
  return(output_text)

# query = input("Describe all you know about covid?" + '\n')
result = chain.invoke("What is best in life?")
print(print_result(result))
import ipdb; ipdb.set_trace()
# -------------------------------------------------------------------------
# template = """Answer the question based on your existing knowledge but with special attention to the following context:
# {context}

# Question: {question}
# """
# prompt = ChatPromptTemplate.from_template(template)
# chat_model = ChatOllama(
#     model="llama2",
# )

# chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | chat_model
#     | StrOutputParser()
# )
# # print(chain.invoke("where did harrison work?"))

# from langchain.schema import format_document
# from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
# from langchain_core.runnables import RunnableParallel
# from langchain.prompts.prompt import PromptTemplate

# _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""
# CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
# template = """Answer the question based only on the following context:
# {context}

# Question: {question}
# """
# ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
# DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


# def _combine_documents(
#     docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
# ):
#     doc_strings = [format_document(doc, document_prompt) for doc in docs]
#     return document_separator.join(doc_strings)

# _inputs = RunnableParallel(
#     standalone_question=RunnablePassthrough.assign(
#         chat_history=lambda x: get_buffer_string(x["chat_history"])
#     )
#     | CONDENSE_QUESTION_PROMPT
#     | ChatOllama(model="llama2",)
#     | StrOutputParser(),
# )
# _context = {
#     "context": itemgetter("standalone_question") | retriever | _combine_documents,
#     "question": lambda x: x["standalone_question"],
# }
# conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOllama(model="llama2",)

# # print(conversational_qa_chain.invoke(
# #     {
# #         "question": "where did harrison work?",
# #         "chat_history": [],
# #     }
# # )
# #       )

# print(conversational_qa_chain.invoke(
#     {
#         "question": "Citing specific ideas (using their path), explain long covid to me?",
#         "chat_history": [
#             HumanMessage(content="Do you understand covid?"),
#             AIMessage(content="Yes, there are many idea documents on this topic."),
#         ],
#     }
# ))
