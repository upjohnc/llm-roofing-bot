from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama

import constants
from vector_store import get_vector_store


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


def get_llm(model: str) -> ChatOllama:
    return ChatOllama(model=model, temperature=0)


def get_llm_response_two_call(query: str) -> str:
    def get_retriever() -> VectorStoreRetriever:
        vector_store = get_vector_store()
        return vector_store.as_retriever(search_kwargs={"k": 2})

    llm = get_llm(constants.MODEL)

    documents = get_retriever().invoke(query)

    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.

            Use the following documents to answer the question and
            only use the documents.

            If you don't know the answer, just say that you don't know.

            Use three sentences maximum and keep the answer concise:
            Question: {question}
            Documents: {documents}
            Answer:
        """,
        input_variables=["question", "documents"],
    )

    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain.invoke({"question": query, "documents": documents})


def get_response_grader(query: str, original_response: str) -> dict:
    prompt = PromptTemplate(
        template="""You are a teacher grading a quiz. You will be given:
        a query and a response to that query

        You are grading RELEVANCE RECALL:
        A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION.
        A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION.
        1 is the highest (best) score. 0 is the lowest score you can give.

        query: {question}
        response to that query: {statement}

        Give a binary score 1 or 0 score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
        """,
        input_variables=["question"],
    )
    llm = get_llm(constants.MODEL)
    response_grader = prompt | llm | JsonOutputParser()
    return response_grader.invoke({"statement": original_response, "question": query})
