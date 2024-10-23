from chromadb import Documents
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

import constants
from vector_store import get_vector_store


def get_end_response(question, documents):
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.

            Use the following documents to answer the question.

            If you don't know the answer, just say that you don't know.

            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
        """,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(model=constants.MODEL, temperature=0)

    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"question": question, "documents": documents})


def get_retriever() -> VectorStoreRetriever:
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": 2})


def get_llm_response_single_call(question: str) -> str:
    documents = get_retriever().invoke(question)

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.

            Use the following documents to answer the question.

            ONLY USE the details given to you in the context/documents when answering.
            if there are not enough details in the document or context to answer
            the question just reply, 'I dont know'.


            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
        """,
        input_variables=["question", "documents"],
    )

    llm = ChatOllama(model=constants.MODEL, temperature=0)

    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"question": question, "documents": documents})
