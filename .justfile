default:
    just --list

install:
    poetry install --sync --no-root --with dev

pre-commit:
    pre-commit install

run *args:
    PYTHONPATH=./src poetry run python src/code.py {{ args }}

ollama-start:
    ollama serve

llama3:
    ollama pull llama3

create-vector-db:
    rm -rf ./chroma_langchain_db
    PYTHONPATH=./src poetry run python src/vector_store.py
