import click

from chat_calls import run_graph_one_call, run_graph_two_call


def simple_call(call_type: str, question_number: int) -> None:
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


def cli_chat():
    print()
    print("I am an information source on roofing and", flush=True)
    print("am here to help you inform yourself for", flush=True)
    print("when you need to re-roof your house.", flush=True)
    print()

    while True:
        print()
        question = input("Roofing Question > ")
        if not question:
            continue
        if question.strip().lower() in ["/quit", "/q"]:
            break

        result = run_graph_two_call(question)
        print()
        print(result["response"])

        if result["grade"]["score"] == 0:
            what = (
                "\n\nYour question may be outside of what I know."
                "\nMay we call you to discuss your question further?"
            )
            print(what)
            call_request = input("(yes or no) > ")
            if call_request.strip().lower() == "yes":
                phone_number = input("what number may we reach you at? > ")
                print(f"Number where to call: {phone_number}")


@click.command
@click.option(
    "--call-type",
    type=click.Choice(["one_call", "two_call", "cli"]),
    help="one_call or two_call",
)
@click.option(
    "--question-number",
    default=1,
    help="number of the question to use\nused in junction with one_call and two_call",
)
def main(call_type, question_number):
    if call_type == "cli":
        _ = cli_chat()
    else:
        _ = simple_call(call_type, question_number)


if __name__ == "__main__":
    main()
