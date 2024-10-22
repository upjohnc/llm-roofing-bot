from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from doc_evaluate import grade_docs_on_roofing
from llm_response import get_end_response, get_llm_response_single_call
from vector_store import get_vetor_store


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
    vector_store = get_vetor_store()
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


def run_graph_two_call(query: str) -> dict:
    workflow = StateGraph(GraphState)
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
