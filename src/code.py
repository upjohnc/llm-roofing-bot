# from complete_response import get_end_response
# from docs_evaluate import grade_docs_for_tavily_search
from langchain_core.vectorstores import VectorStoreRetriever

# from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from vector_store import create_vector_store, get_split_docs


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
    web_search: bool
    documents: list[str]
    steps: list[str]


def get_vector_store(state: GraphState) -> dict:
    state["steps"].append("get_vector_store")
    vector_store = create_vector_store(get_split_docs())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    return {"query": state["query"], "retriever": retriever, "steps": state["steps"]}


def run():
    vector_store = create_vector_store(get_split_docs())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    print(retriever)


if __name__ == "__main__":
    run()
