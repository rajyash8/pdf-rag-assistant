from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def create_database(pdf_path):

    # -----------------------------
    # 1. Load PDF
    # -----------------------------

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # -----------------------------
    # 2. Split PDF into chunks
    # -----------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Loaded pages: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")

    # -----------------------------
    # 3. Local HuggingFace embeddings
    # -----------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -----------------------------
    # 4. Chroma vector database
    # -----------------------------

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="pdf_documents"
    )

    return vectorstore
