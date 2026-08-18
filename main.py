import os
import tempfile

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from createdatabase import create_database


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# -----------------------------
# Mistral API Key
# -----------------------------

if "MISTRAL_API_KEY" not in st.secrets:
    st.error("MISTRAL_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]


# -----------------------------
# Page UI
# -----------------------------

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions based only on its contents."
)


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# -----------------------------
# Session State
# -----------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# -----------------------------
# Process PDF
# -----------------------------

if uploaded_file is not None:

    # Only create database when a NEW PDF is uploaded
    if st.session_state.pdf_name != uploaded_file.name:

        with st.spinner("Processing PDF..."):

            temp_path = None

            try:

                # Create temporary PDF file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(uploaded_file.getbuffer())
                    temp_path = temp_file.name

                # Create in-memory Chroma database
                vectorstore = create_database(temp_path)

                # Save vectorstore in Streamlit session
                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = uploaded_file.name

                st.success("PDF processed successfully!")

            except Exception as e:

                st.error(
                    f"Error while processing PDF: {str(e)}"
                )

                st.session_state.vectorstore = None
                st.session_state.pdf_name = None

            finally:

                # Remove temporary PDF
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# -----------------------------
# Ask Questions
# -----------------------------

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:",
        placeholder="What is this PDF about?"
    )

    if question:

        with st.spinner("Searching the PDF..."):

            try:

                # MMR retrieval
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                # Retrieve relevant chunks
                documents = retriever.invoke(question)

                # Combine retrieved context
                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )

                # Mistral chat model
                llm = ChatMistralAI(
                    model="mistral-small-latest",
                    temperature=0
                )

                # Prompt
                prompt = ChatPromptTemplate.from_template(
                    """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:
"I couldn't find that information in the PDF."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
                )

                # Create chain
                chain = prompt | llm

                # Generate answer
                response = chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )

                st.subheader("Answer")

                st.write(response.content)

                # Optional source information
                with st.expander("📄 View Sources"):

                    for i, document in enumerate(
                        documents,
                        start=1
                    ):

                        page_number = (
                            document.metadata.get("page", "Unknown")
                        )

                        st.markdown(
                            f"**Source {i} — Page {page_number + 1 if isinstance(page_number, int) else page_number}**"
                        )

                        st.write(
                            document.page_content
                        )

                        st.divider()

            except Exception as e:

                st.error(
                    f"Error while answering question: {str(e)}"
                )
