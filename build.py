
#!/usr/bin/env python3

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import ObsidianLoader
# from obsidian import ObsidianLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = ObsidianLoader('/mnt/d/Dropbox/Tasks')
docs = loader.load()

# print(docs[0])
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents, embedding=OllamaEmbeddings())
vectorstore.save_local("tasks_faiss_index")
