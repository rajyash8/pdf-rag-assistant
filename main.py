import os
import hashlib
import tempfile

import streamlit as st

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

from createdatabase import create_database


# =====================================================
# STREAMLIT CONFIG
# =====================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# =====================================================
# MISTRAL API KEY
# =====================================================

if "MISTRAL_API_KEY" not in st.secrets:

    st.error(
        "MISTRAL_API_KEY is not configured in Streamlit Secrets."
    )

    st.stop()


MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]


# =====================================================
# MISTRAL LLM
# =====================================================

@st.cache_resource
def get_llm():

    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=MISTRAL_API_KEY
    )


# =====================================================
# SESSION STATE
# =====================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None


if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# =====================================================
# TITLE
# =====================================================

st.title("📚 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions based only on its contents."
)


# =====================================================
# PDF UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# =====================================================
# PROCESS PDF
# =====================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    # Create unique hash for the uploaded PDF
    current_hash = hashlib.md5(pdf_bytes).hexdigest()


    # Only process when a new PDF is uploaded
    if st.session_state.pdf_hash != current_hash:

        with st.spinner(
            "Processing PDF... This may take a moment."
        ):

            temp_path = None

            try:

                # -----------------------------
                # Create temporary PDF
                # -----------------------------

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(pdf_bytes)

                    temp_path = temp_file.name


                # -----------------------------
                # Create vector database
                # -----------------------------

                vectorstore = create_database(
                    temp_path
                )


                # -----------------------------
                # Save in session
                # -----------------------------

                st.session_state.vectorstore = vectorstore

                st.session_state.pdf_hash = current_hash

                st.session_state.pdf_name = uploaded_file.name


                st.success(
                    f"✅ {uploaded_file.name} processed successfully!"
                )


            except Exception as e:

                st.error(
                    f"❌ Error while processing PDF: {str(e)}"
                )

                st.session_state.vectorstore = None

                st.session_state.pdf_hash = None

                st.session_state.pdf_name = None


            finally:

                # -----------------------------
                # Delete temporary PDF
                # -----------------------------

                if (
                    temp_path
                    and os.path.exists(temp_path)
                ):

                    os.remove(temp_path)


# =====================================================
# ASK QUESTION
# =====================================================

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask a question")


    # =================================================
    # FORM
    # =================================================

    with st.form("question_form"):

        question = st.text_input(
            "Enter your question:",
            placeholder="What is this PDF about?"
        )


        submitted = st.form_submit_button(
            "Ask"
        )


    # =================================================
    # RUN RAG
    # =================================================

    if submitted and question.strip():

        with st.spinner(
            "Searching the PDF..."
        ):

            try:

                # -------------------------------------
                # 1. Create retriever
                # -------------------------------------

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


                # -------------------------------------
                # 2. Search PDF
                # -------------------------------------

                documents = retriever.invoke(
                    question
                )


                # -------------------------------------
                # 3. Build context
                # -------------------------------------

                context = "\n\n".join(
                    document.page_content
                    for document in documents
                )


                # -------------------------------------
                # 4. Get Mistral
                # -------------------------------------

                llm = get_llm()


                # -------------------------------------
                # 5. Prompt
                # -------------------------------------

                prompt = ChatPromptTemplate.from_template(
                    """
You are a helpful PDF question-answering assistant.

Your job is to answer the user's question using
ONLY the information provided in the context.

Rules:

1. Do not use outside knowledge.
2. Do not make up information.
3. If the answer is not present in the context,
   say exactly:

"I couldn't find that information in the PDF."

4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
                )


                # -------------------------------------
                # 6. Create chain
                # -------------------------------------

                chain = prompt | llm


                # -------------------------------------
                # 7. Ask Mistral
                # -------------------------------------

                response = chain.invoke(
                    {
                        "context": context,
                        "question": question
                    }
                )


                # -------------------------------------
                # 8. Display answer
                # -------------------------------------

                st.subheader("🤖 Answer")

                st.write(
                    response.content
                )


                # -------------------------------------
                # 9. Display sources
                # -------------------------------------

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

                error_message = str(e)


                # -------------------------------------
                # Rate limit error
                # -------------------------------------

                if (
                    "429" in error_message
                    or "rate_limit" in error_message.lower()
                    or "rate limit" in error_message.lower()
                ):

                    st.error(
                        """
❌ Mistral API rate limit reached.

Your PDF embeddings are running locally,
so this error is coming from the Mistral
chat request.

Please wait for the API limit to reset or
use an account/plan with available API capacity.
"""
                    )


                # -------------------------------------
                # Other error
                # -------------------------------------

                else:

                    st.error(
                        f"❌ Error while answering question: "
                        f"{error_message}"
                    )
