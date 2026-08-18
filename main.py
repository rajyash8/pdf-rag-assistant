import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import (
    ChatMistralAI,
    MistralAIEmbeddings
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from createdatabase import create_database


load_dotenv()


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="🤖"
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its contents."
)


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

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            # ---------------------------------------------
            # Remove old Chroma database
            # ---------------------------------------------

            if os.path.exists("chroma_db"):
                shutil.rmtree("chroma_db")

            # ---------------------------------------------
            # Create temporary PDF
            # ---------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getvalue()
                )

                pdf_path = temp_file.name

            try:

                # -----------------------------------------
                # Create new Chroma database
                # -----------------------------------------

                create_database(pdf_path)

            finally:

                # -----------------------------------------
                # Delete temporary PDF
                # -----------------------------------------

                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

        st.success(
            "PDF processed successfully! ✅"
        )

        # Force Streamlit to rerun cleanly
        st.rerun()


# =========================================================
# ASK QUESTIONS
# =========================================================

if os.path.exists("chroma_db"):

    st.divider()

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question:"
    )

    if question:

        with st.spinner("Searching the document..."):

            # -------------------------------------------------
            # Embedding model
            # -------------------------------------------------

            embedding_model = MistralAIEmbeddings(
                model="mistral-embed"
            )


            # -------------------------------------------------
            # Load Chroma
            # -------------------------------------------------

            vectorstore = Chroma(
                persist_directory="chroma_db",
                collection_name="pdf_documents",
                embedding_function=embedding_model
            )


            # -------------------------------------------------
            # Retriever
            # -------------------------------------------------

            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 4,
                    "fetch_k": 10,
                    "lambda_mult": 0.5
                }
            )


            # -------------------------------------------------
            # Retrieve relevant chunks
            # -------------------------------------------------

            retrieved_docs = retriever.invoke(
                question
            )


            # -------------------------------------------------
            # Create context
            # -------------------------------------------------

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )


        # =====================================================
        # MISTRAL LLM
        # =====================================================

        llm = ChatMistralAI(
            model="mistral-small-latest"
        )


        # =====================================================
        # PROMPT
        # =====================================================

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful AI assistant.

Answer the user's question using only the
provided context.

If the answer cannot be found in the context,
say you don't know.

Context:
{data}"""
            ),
            (
                "human",
                "{input}"
            )
        ])


        # =====================================================
        # SEND QUESTION + CONTEXT TO LLM
        # =====================================================

        messages = prompt.invoke({
            "input": question,
            "data": context
        })

        response = llm.invoke(messages)


        # =====================================================
        # DISPLAY ANSWER
        # =====================================================

        st.subheader("Answer")

        st.write(response.content)