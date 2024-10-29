from pathlib import Path
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import constants


def get_split_docs() -> list[Document]:
    file_path = Path(__file__).parent / "support" / "roofing.txt"
    with open(file_path, "r") as f:
        roofing_docs = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=constants.CHUNK_SIZE,
        chunk_overlap=20,
        separators=["\n\n", "\n", " "],
        length_function=lambda text: len(text),
        is_separator_regex=False,
    )
    doc_list = text_splitter.create_documents([roofing_docs])
    split_docs = text_splitter.split_documents(doc_list)

    return split_docs


def get_vector_store() -> VectorStore:
    embeddings = OllamaEmbeddings(model=constants.MODEL)
    return Chroma(
        collection_name="roof_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )


def create_chroma_db():
    documents = get_split_docs()
    uuids = [str(uuid4()) for _ in documents]

    vector_store = get_vector_store()
    vector_store.add_documents(documents=documents, ids=uuids)

    print("complete")


if __name__ == "__main__":
    create_chroma_db()
