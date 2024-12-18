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
    grade = 1
    if "don't know" in response.lower() or "do not know" in response.lower():
        _ = save_unanswered_questions(query)
        grade = 0

    return dict(response=response, grade=grade)


def run_graph_two_call(query: str) -> dict:

    response = get_llm_response_two_call(query)

    grade = get_response_grader(query, response)

    if grade.get("score") == 1:
        final_response = response
    else:
        _ = save_unanswered_questions(query)
        final_response = (
            "My knowledge base is limited roofs and the process of re-roofing."
            "\nI am happy to share this knowledge with you and help to make informed"
            "\ndecisions."
        )

    return dict(
        response=final_response,
        grade=grade.get("score"),
    )
