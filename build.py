#!/usr/bin/env python3

from operator import itemgetter

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import ObsidianLoader

loader = ObsidianLoader("/mnt/d/Dropbox/Tasks")
docs = loader.load()

# print(docs[0])
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(documents, embedding=OllamaEmbeddings())
vectorstore.save_local("tasks_faiss_index")
