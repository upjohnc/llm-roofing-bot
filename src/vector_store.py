from pathlib import Path
from uuid import uuid4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import constants


def get_pdf_split_docs(dir: str) -> list[Document]:
    pdf_folder_path = Path(__file__).parent.parent / "roofing_docs" / dir
    documents = []
    for file in pdf_folder_path.glob("**/*.pdf"):
        doc_text = PyPDFLoader(str(file.resolve())).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=constants.CHUNK_SIZE,
            chunk_overlap=20,
            separators=["\n\n", "\n", " "],
            length_function=lambda text: len(text),
            is_separator_regex=False,
        )
        split_docs = text_splitter.split_documents(doc_text)
        documents.extend(split_docs)

    return documents


def get_split_docs(dir: str) -> list[Document]:
    file_path = Path(__file__).parent.parent / "roofing_docs" / dir

    documents = []
    for file in file_path.glob("**/*"):
        with open(file, "r") as f:
            doc_text = f.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=constants.CHUNK_SIZE,
            chunk_overlap=20,
            separators=["\n\n", "\n", " "],
            length_function=lambda text: len(text),
            is_separator_regex=False,
        )
        doc_list = text_splitter.create_documents([doc_text])
        split_docs = text_splitter.split_documents(doc_list)
        documents.extend(split_docs)
    return documents


def get_vector_store() -> VectorStore:
    embeddings = OllamaEmbeddings(model=constants.EMBEDDING_MODEL)
    return Chroma(
        collection_name="roof_docs",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",
    )


def add_vector_docs(documents: list[Document]):
    uuids = [str(uuid4()) for _ in documents]

    vector_store = get_vector_store()
    vector_store.add_documents(documents=documents, ids=uuids)


def create_chroma_db():
    documents = get_pdf_split_docs("owens-corning")
    _ = add_vector_docs(documents)

    for dir in ["tamko", "gaf"]:
        documents = get_split_docs(dir)
        _ = add_vector_docs(documents)

    print("complete")


if __name__ == "__main__":
    create_chroma_db()
