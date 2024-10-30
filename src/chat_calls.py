import json
from datetime import datetime
from pathlib import Path

from llm_response import (
    get_llm_response_single_call,
    get_llm_response_two_call,
    get_response_grader,
)


def save_unanswered_questions(query: str) -> None:
    dir = Path(__file__).parent / "unanswered_query"
    if not dir.exists():
        dir.mkdir()
    unanswered_file = dir / "queries.json"

    unanswered_data = {
        "query": query,
        "timestamp_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%m:%s"),
    }
    with open(unanswered_file, "a") as f:
        f.write(json.dumps(unanswered_data))
        f.write("\n")


def run_graph_one_call(query: str):
    response = get_llm_response_single_call(query)
    if "don't know" in response.lower():
        _ = save_unanswered_questions(query)

    print(response)

    return 1


def run_graph_two_call(query: str) -> dict:

    response = get_llm_response_two_call(query)

    grade = get_response_grader(query, response)

    if grade.get("score") == 1:
        final_response = response
    else:
        _ = save_unanswered_questions(query)
        final_response = (
            "I do not know the answer to the question.  May we call you to discuss?"
        )

    return dict(
        response=final_response,
        grade=grade,
    )
