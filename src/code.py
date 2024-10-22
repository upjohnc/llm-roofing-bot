import sys

from lang_graph import run_graph_one_call, run_graph_two_call

if __name__ == "__main__":
    questions = {
        "1": "what is a roof?",
        "2": "what materials are in a roof?",
        "3": "what is the weather today?",
        "4": "what should I pay for a roof?",
        "5": "how much should I pay?",
        "6": "what are roofing terms?",
    }
    graphs = {"one_call": run_graph_one_call, "two_call": run_graph_two_call}

    args = sys.argv

    graph_type = "one_call" if len(args) < 2 else args[1]
    graph_function = graphs[graph_type]
    index_question = "1" if len(args) < 3 else args[2]

    question = questions[index_question]

    result = graph_function(question)
    print(result)
