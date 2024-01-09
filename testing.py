#!/usr/bin/env python3
#
# from llama_index.llms import Ollama
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama

documents = SimpleDirectoryReader("data").load_data()

embedding_llm = LangchainEmbedding(langchain_embeddings=OllamaEmbeddings())
service_context = ServiceContext.from_defaults(llm=ChatOllama(model="llama2"), embed_model=embedding_llm,)

index = VectorStoreIndex.from_documents(documents, service_context=service_context)

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
custom_chat_history = [
    ChatMessage(
        role=MessageRole.USER,
        content="Hello assistant, we are having a insightful discussion about Paul Graham today.",
    ),
    ChatMessage(role=MessageRole.ASSISTANT, content="Okay, sounds good."),
]
messages = []
# messages.append({"role": "user", "content": prompt})
# index = VectorStoreIndex.from_documents(documents)
prompt = "What is McCarthy's main thematic interest?" # input("Well?\n")
messages.append({"role": "user", "content": prompt})
response = chat_engine.chat(prompt, chat_history=custom_chat_history)
print(response)
message = {"role": "assistant", "content": response.response}
messages.append(message) # Add response to message history

# query_engine = index.as_query_engine()
# response = query_engine.query("What did the author do growing up?")
# print(response)
