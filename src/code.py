import click

from chat_calls import run_graph_one_call, run_graph_two_call


@click.command
@click.option("--call-type", help="one_call or two_call")
@click.option("--question-number", default=1, help="number of the question to use")
def main(call_type, question_number):
    questions = {
        1: "what is a roof?",
        2: "what materials are in a roof?",
        3: "what is the weather today?",
        4: "what should I pay for a roof?",
        5: "how much should I pay?",
        6: "what are roofing terms?",
    }
    graphs = {"one_call": run_graph_one_call, "two_call": run_graph_two_call}

    graph_function = graphs[call_type]

    question = questions[question_number]

    result = graph_function(question)
    print(result)


if __name__ == "__main__":
    main()
