from langchain_core.output_parsers import JsonOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from doc_evaluate import grade_docs_on_roofing
from llm_response import get_end_response, get_llm_response_single_call
from vector_store import get_vector_store


class GraphState(TypedDict):
    """
    State of lang graph

    Attributes:
        query: query
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    query: str
    generation: str
    retriever: VectorStoreRetriever
    enough_information: bool
    documents: list[str]
    steps: list[str]


def get_vector_retriever(state: GraphState) -> dict:
    state["steps"].append("get_vector_store")
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return {"query": state["query"], "retriever": retriever, "steps": state["steps"]}


def check_doc_grade(state: GraphState) -> dict:
    state["steps"].append("check_doc_grade")
    (enough_info, docs) = grade_docs_on_roofing(state["retriever"], state["query"])
    return {
        "query": state["query"],
        "retriever": state["retriever"],
        "documents": docs,
        "enough_information": enough_info,
        "steps": state["steps"],
    }


def decide_to_generate(state: GraphState) -> str:
    if state["enough_information"] is False:
        return "end_conversation"
    return "generate"


def generate(state: GraphState) -> dict:
    state["steps"].append("generate")
    response = get_end_response(state["query"], state["documents"])
    return {
        "query": state["query"],
        "retriever": state["retriever"],
        "documents": state["documents"],
        "enough_information": state["enough_information"],
        "generation": response,
        "steps": state["steps"],
    }


def end_conversation(state: GraphState) -> dict:
    return {
        "query": state["query"],
        "retriever": state["retriever"],
        "documents": state["documents"],
        "generation": "I don't know",
        "steps": state["steps"],
    }


def run_graph_one_call(query: str):
    print(get_llm_response_single_call(query))

    return 1


def run_graph_two_call(query: str) -> dict:
    from chromadb import Documents
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.vectorstores import VectorStoreRetriever
    from langchain_ollama import ChatOllama

    import constants
    from vector_store import get_vector_store

    def get_retriever() -> VectorStoreRetriever:
        vector_store = get_vector_store()
        return vector_store.as_retriever(search_kwargs={"k": 2})

    llm = ChatOllama(model=constants.MODEL, temperature=0)
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
    response = rag_chain.invoke({"question": query, "documents": documents})

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
    retrieval_grader = prompt | llm | JsonOutputParser()
    grade = retrieval_grader.invoke({"statement": response, "question": query})

    final_response = (
        "I do not know the answer to the question.  May we call you to discuss?"
    )
    if grade.get("score") == 1:
        final_response = response

    return dict(
        response=final_response,
        grade=grade,
    )


def run_graph_two_call_hold(query: str) -> dict:
    workflow = StateGraph(GraphState)
    # make first call
    # store response
    # ask if repsonse is good enough
    workflow.add_node("vector_retriever", get_vector_retriever)
    workflow.add_node("doc_grade", check_doc_grade)
    workflow.add_node("end_conversation", end_conversation)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "vector_retriever")
    workflow.add_edge("vector_retriever", "doc_grade")

    workflow.add_conditional_edges(
        "doc_grade",
        decide_to_generate,
        {"end_conversation": "end_conversation", "generate": "generate"},
    )

    workflow.add_edge("end_conversation", END)
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()

    state_dict = custom_graph.invoke({"query": query, "steps": []})
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}
