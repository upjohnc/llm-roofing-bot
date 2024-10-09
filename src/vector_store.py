from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document

import constants


def get_split_docs() -> list[Document]:
    file_path = Path(__file__).parent.resolve() / "support" / "roofing.txt"
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


def create_vector_store(docs: list[Document]) -> VectorStore:
    embeddings = OllamaEmbeddings(model=constants.MODEL)
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store
