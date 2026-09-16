import hashlib
import tempfile
import os

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# =========================================================
# PAGE TITLE
# =========================================================

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions based only on its contents."
)


# =========================================================
# LOCAL EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


# =========================================================
# LOCAL LLM - TINYLLAMA
# =========================================================

@st.cache_resource
def get_llm():

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.05
    )

    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    return llm


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:

    st.session_state.vectorstore = None


if "pdf_hash" not in st.session_state:

    st.session_state.pdf_hash = None


if "pdf_name" not in st.session_state:

    st.session_state.pdf_name = None


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()


    # Only process a new PDF
    if st.session_state.pdf_hash != current_hash:

        with st.spinner(
            "Processing PDF... Please wait."
        ):

            temp_path = None

            try:

                # -----------------------------------------
                # Save PDF temporarily
                # -----------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(
                        pdf_bytes
                    )

                    temp_path = temp_file.name


                # -----------------------------------------
                # Load PDF
                # -----------------------------------------

                loader = PyPDFLoader(
                    temp_path
                )

                documents = loader.load()


                # -----------------------------------------
                # Split PDF
                # -----------------------------------------

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = text_splitter.split_documents(
                    documents
                )


                # -----------------------------------------
                # Local embeddings
                # -----------------------------------------

                embeddings = get_embeddings()


                # -----------------------------------------
                # Create Chroma database
                # -----------------------------------------

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="pdf_documents"
                )


                # -----------------------------------------
                # Save in session
                # -----------------------------------------

                st.session_state.vectorstore = vectorstore

                st.session_state.pdf_hash = current_hash

                st.session_state.pdf_name = uploaded_file.name


                st.success(
                    f"✅ {uploaded_file.name} processed successfully!"
                )

                st.info(
                    f"📄 Pages: {len(documents)} | "
                    f"Chunks: {len(chunks)}"
                )


            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF:\n\n{str(e)}"
                )

                st.session_state.vectorstore = None

                st.session_state.pdf_hash = None

                st.session_state.pdf_name = None


            finally:

                if (
                    temp_path is not None
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# =========================================================
# QUESTION SECTION
# =========================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask a question")


    # =====================================================
    # QUESTION FORM
    # =====================================================

    with st.form("question_form"):

        question = st.text_input(
            "Enter your question:",
            placeholder="What is this PDF about?"
        )

        submitted = st.form_submit_button(
            "Ask"
        )


    # =====================================================
    # ANSWER
    # =====================================================

    if submitted and question.strip():

        with st.spinner(
            "Searching the PDF and generating answer..."
        ):

            try:

                # -----------------------------------------
                # Retriever
                # -----------------------------------------

                retriever = (
                    st.session_state.vectorstore
                    .as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )
                )


                # -----------------------------------------
                # Retrieve relevant documents
                # -----------------------------------------

                documents = retriever.invoke(
                    question
                )


                # -----------------------------------------
                # Build context
                # -----------------------------------------

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )


                # -----------------------------------------
                # Local TinyLlama
                # -----------------------------------------

                llm = get_llm()


                # -----------------------------------------
                # Prompt
                # -----------------------------------------

                prompt = f"""
You are a PDF question answering assistant.

Answer the question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the answer is not available in the context,
say:

"I couldn't find that information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


                # -----------------------------------------
                # Generate answer
                # -----------------------------------------

                response = llm.invoke(
                    prompt
                )


                # -----------------------------------------
                # Display answer
                # -----------------------------------------

                st.subheader("🤖 Answer")

                st.write(
                    response
                )


                # -----------------------------------------
                # Sources
                # -----------------------------------------

                with st.expander(
                    "📄 View Sources"
                ):

                    for i, document in enumerate(
                        documents,
                        start=1
                    ):

                        page_number = document.metadata.get(
                            "page",
                            "Unknown"
                        )


                        if isinstance(
                            page_number,
                            int
                        ):

                            page_number += 1


                        st.markdown(
                            f"**Source {i} — Page {page_number}**"
                        )


                        st.write(
                            document.page_content
                        )


                        st.divider()


            except Exception as e:

                st.error(
                    f"❌ Error while answering question:\n\n{str(e)}"
                )
