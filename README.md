# Roofing Chat Bot

A friend asked me if I could write a chatbot for a roofing company.
The bot would be an informative tool for people landing on the page
and would like to be informed about roofing.  The ideal is that
as an individual asks questions about what he/she should know about
replacing a roof, then the person comes to trust the roofing company and
the person becomes a prospective client.

## Design

The design is to minimize false positive responses from the llm.
The code and prompts are written to keep the answer focused
on the knowledge base in the vector store.

At the point the bot cannot answer a question then the
response to the user is the message that the roofing company
could call the user to follow up.

The chatbot does not retain history.  The thinking is that the bot
would be answering questions about roofing and wouldn't need
chat history to make better answers.

### Two Different Chat Flows

I set up two flows.  `one_call` and `two_call`.
The `one_call` includes in the prompt the check that
only the documents are being used to answer the query.
In the `two_call` there is a grader of the answer
to the query.

I wrote both to get feedback.  I would like to hear what
people think is the best solution.

### Save unknown queries

A feature to add to the code is saving queries that aren't answered.
The list of unanswered queries can be reviewed later and then
information added to the vector store.

## Usage

The llm model is `llama3.2` and the embedding model is `mxbai-embed-large`.
You can have the ollama running locally with `llama3.2` and `mxbai-embed-large` loaded.
Instructions can be found at the ollama site `https://ollama.com/`.

The project runs with poetry package manager.  You can install
it locally with `pip install poetry`.

In the `.justfile`, there are common commands.  You can install
the tool `just` with `brew install just`.  [link to the just github](https://github.com/casey/just)

With `just` and `poetry` installed, you can set up the python
virtual env with `just install`.  Then you can make a call
to the python script.  `just run --help` will show the cli
scripts help docs.
