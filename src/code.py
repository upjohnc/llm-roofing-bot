import sys

from lang_graph import run_graph

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
