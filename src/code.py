import sys

from doc_evaluate import grade_docs_on_roofing
from lang_graph import run_graph
from llm_response import get_end_response
from vector_store import create_vector_store, get_split_docs


def run(question: str):
    vector_store = create_vector_store(get_split_docs())
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    enough, docs = grade_docs_on_roofing(retriever, question)

    if enough:
        response = get_end_response(question, documents=docs)
    else:
        response = "I do not know"
    print(response)


if __name__ == "__main__":
    questions = {
        "1": "what is a roof",
        "2": "what materials are in a roof",
        "3": "what is the weather today",
        "4": "what should I pay for a roof",
        "5": "how much should I pay",
    }

    args = sys.argv
    index_question = "1" if len(args) == 1 else args[1]

    question = questions[index_question]

    result = run_graph(question)
    print(result)
