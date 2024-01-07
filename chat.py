#!/usr/bin/env python3

from operator import itemgetter

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_citation_fuzzy_match_chain
from langchain_community.chat_models import ChatOllama
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
from langchain.chains import create_citation_fuzzy_match_chain

vectorstore = FAISS.load_local("tasks_faiss_index", embeddings=OllamaEmbeddings())
retriever = vectorstore.as_retriever()

# llm=Cohere(model="command")
model = ChatOllama(model="llama2",)

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


from operator import itemgetter

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

# First we add a step to load memory
# This adds a "memory" key to the input object
loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
)
# Now we calculate the standalone question
standalone_question = {
    "standalone_question": {
        "question": lambda x: x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"]),
    }
    | CONDENSE_QUESTION_PROMPT
    | model
    | StrOutputParser(),
}
# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | model,
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

inputs = {"question": "Tell me about covid 19 vaccines"}
result = final_chain.invoke(inputs)
print(result)


# # Note that the memory does not save automatically
# # This will be improved in the future
# # For now you need to save it yourself
# memory.save_context(inputs, {"answer": result["answer"].content})

# memory.load_memory_variables({})

# {'history': [HumanMessage(content='where did harrison work?'),
#   AIMessage(content='Harrison was employed at Kensho.')]}

# inputs = {"question": "but where did he really work?"}
# result = final_chain.invoke(inputs)
# result

# {'answer': AIMessage(content='Harrison actually worked at Kensho.'),
#  'docs': [Document(page_content='harrison worked at kensho')]}
import ipdb; ipdb.set_trace()
