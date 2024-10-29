from llm_response import (
    get_llm_response_single_call,
    get_llm_response_two_call,
    get_response_grader,
)


def run_graph_one_call(query: str):
    print(get_llm_response_single_call(query))

    return 1


def run_graph_two_call(query: str) -> dict:

    response = get_llm_response_two_call(query)

    grade = get_response_grader(query, response)

    final_response = (
        "I do not know the answer to the question.  May we call you to discuss?"
    )
    if grade.get("score") == 1:
        final_response = response

    return dict(
        response=final_response,
        grade=grade,
    )
