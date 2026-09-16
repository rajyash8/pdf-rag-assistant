import os
import tempfile
import hashlib

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from createdatabase import create_database


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# MISTRAL API KEY
# ============================================================

if "MISTRAL_API_KEY" not in st.secrets:
    st.error("MISTRAL_API_KEY is not configured in Streamlit Secrets.")
    st.stop()

os.environ["MISTRAL_API_KEY"] = st.secrets["MISTRAL_API_KEY"]


# ============================================================
# PAGE UI
# ============================================================

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions based only on its contents."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    # Read uploaded PDF
    pdf_bytes = uploaded_file.getvalue()

    # Generate unique hash for PDF
    current_pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

    # Only process if this is a different PDF
    if st.session_state.pdf_hash != current_pdf_hash:

        with st.spinner("Processing PDF..."):

            temp_path = None

            try:

                # ---------------------------------------------
                # Create temporary PDF
                # ---------------------------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(pdf_bytes)
                    temp_path = temp_file.name

                # ---------------------------------------------
                # Create vector database
                # ---------------------------------------------

                vectorstore = create_database(temp_path)

                # ---------------------------------------------
                # Save in session
                # ---------------------------------------------

                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_hash = current_pdf_hash

                st.success(
                    f"✅ {uploaded_file.name} processed successfully!"
                )

            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF: {str(e)}"
                )

                st.session_state.vectorstore = None
                st.session_state.pdf_hash = None

            finally:

                # ---------------------------------------------
                # Delete temporary file
                # ---------------------------------------------

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)


# ============================================================
# ASK QUESTIONS
# ============================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask a question")

    question = st.text_input(
        "Enter your question:",
        placeholder="What is this PDF about?"
    )

    if question:

        with st.spinner("Searching the PDF..."):

            try:

                # =================================================
                # RETRIEVER
                # =================================================

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 4,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                # =================================================
                # RETRIEVE DOCUMENTS
                # =================================================

                documents = retriever.invoke(question)

                # =================================================
                # BUILD CONTEXT
                # =================================================

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )

                # =================================================
                # MISTRAL LLM
                # =================================================

                llm = ChatMistralAI(
                    model="mistral-small-latest",
                    temperature=0,
                    api_key=os.environ["MISTRAL_API_KEY"]
                )

                # =================================================
                # PROMPT
                # =================================================

                prompt = ChatPromptTemplate.from_template(
                    """
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer cannot be found in the context, say:

"I couldn't find that information in the PDF."

Rules:
- Do not use outside knowledge.
- Do not make up information.
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
                )

                # =================================================
                # CHAIN
                # =================================================

                chain = prompt | llm

                # =================================================
                # GENERATE ANSWER
                # =================================================

                response = chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )

                # =================================================
                # DISPLAY ANSWER
                # =================================================

                st.subheader("Answer")

                st.write(response.content)

                # =================================================
                # SOURCES
                # =================================================

                with st.expander("📄 View Sources"):

                    for i, document in enumerate(
                        documents,
                        start=1
                    ):

                        page_number = document.metadata.get(
                            "page",
                            "Unknown"
                        )

                        if isinstance(page_number, int):
                            page_display = page_number + 1
                        else:
                            page_display = page_number

                        st.markdown(
                            f"**Source {i} — Page {page_display}**"
                        )

                        st.write(
                            document.page_content
                        )

                        st.divider()

            except Exception as e:

                st.error(
                    f"❌ Error while answering question: {str(e)}"
                )
