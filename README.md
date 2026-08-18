# 🤖 PDF RAG Assistant

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its contents.

The application processes the uploaded document, splits it into smaller chunks, generates embeddings using **Mistral**, stores them in a **Chroma vector database**, retrieves the most relevant information using an MMR-based retriever, and uses **Mistral Small** to generate the final answer.

🚀 **Deployed with Streamlit**

---

## 📌 Features

* 📄 Upload PDF documents directly through the UI
* ✂️ Automatically split documents into smaller chunks
* 🧠 Generate embeddings using Mistral
* 🗄️ Store document embeddings in ChromaDB
* 🔎 Retrieve relevant document chunks using MMR
* 🤖 Generate answers using Mistral Small
* 💬 Ask multiple questions about the uploaded document
* 🌐 Streamlit-based web interface
* 🚫 Reduces hallucinations by instructing the model to answer using the retrieved context

---

## 🏗️ RAG Architecture

```text
                 📄 Upload PDF
                       │
                       ▼
                 PyPDFLoader
                       │
                       ▼
                Text Chunking
                       │
                       ▼
              Mistral Embeddings
                       │
                       ▼
                Chroma Vector DB
                       │
                       ▼
                 MMR Retriever
                       │
             ┌─────────┴─────────┐
             │                   │
      User Question        Relevant Chunks
             │                   │
             └─────────┬─────────┘
                       ▼
                Prompt Template
                       │
                       ▼
             Mistral Small LLM
                       │
                       ▼
                  💬 Answer
```

---

## 🛠️ Tech Stack

| Technology                     | Purpose                         |
| ------------------------------ | ------------------------------- |
| Python                         | Programming language            |
| Streamlit                      | Web application UI              |
| LangChain                      | RAG pipeline and integrations   |
| Mistral AI                     | Embeddings and LLM              |
| ChromaDB                       | Vector database                 |
| PyPDF                          | PDF document loading            |
| RecursiveCharacterTextSplitter | Document chunking               |
| dotenv                         | Environment variable management |

---

## 📂 Project Structure

```text
RAG PROJECT/
│
├── main.py
├── createdatabase.py
├── requirements.txt
├── .env
├── .gitignore
└── chroma_db/
```

### `createdatabase.py`

Responsible for the document ingestion pipeline:

```text
PDF
 ↓
Load document
 ↓
Split into chunks
 ↓
Generate Mistral embeddings
 ↓
Store in ChromaDB
```

### `main.py`

Responsible for the Streamlit application:

```text
Upload PDF
 ↓
Process PDF
 ↓
Retrieve relevant chunks
 ↓
Send context + question to Mistral
 ↓
Display answer
```

---

## ⚙️ How It Works

### 1. Upload a PDF

The user uploads a PDF through the Streamlit interface.

### 2. Document Loading

`PyPDFLoader` extracts the text from the PDF.

### 3. Chunking

The extracted document is divided into smaller chunks using `RecursiveCharacterTextSplitter`.

The project uses:

```python
chunk_size = 1000
chunk_overlap = 200
```

Chunking allows the system to efficiently search large documents.

### 4. Embeddings

Each chunk is converted into a numerical representation using the Mistral embedding model:

```text
mistral-embed
```

### 5. Vector Database

The embeddings and their corresponding document content are stored in **ChromaDB**.

### 6. Retrieval

When the user asks a question, the question is compared against the stored embeddings.

The project uses an **MMR (Maximal Marginal Relevance)** retriever:

```python
search_type="mmr"

k=4
fetch_k=10
lambda_mult=0.5
```

This retrieves relevant chunks while trying to avoid returning highly repetitive information.

### 7. LLM Generation

The retrieved chunks are provided to:

```text
mistral-small-latest
```

along with the user's question.

The model generates the final answer using the retrieved document context.

---

## 🚀 Running the Project Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG-PROJECT
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Mistral API key

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key
```

### 5. Run the application

```bash
streamlit run main.py
```

The application will open in your browser.

---

## 💡 Example Questions

After uploading a PDF, you can ask questions such as:

```text
What is this document about?

What are the main topics discussed?

Summarize the document.

What questions were asked about Operating Systems?

What DSA questions are mentioned?

What SQL questions are present in the document?

Which companies asked questions about OOPs?
```

The answer is generated using information retrieved from the uploaded PDF.

---

## 🔐 Environment Variables

The project requires a Mistral API key.

```env
MISTRAL_API_KEY=your_api_key
```

Do **not** commit your `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
.venv/
__pycache__/
```

---

## 🎯 Why RAG?

A normal LLM may not have access to the contents of a user's private PDF.

RAG solves this by retrieving relevant information from the document and providing that information to the LLM before generating the answer.

Instead of:

```text
Question → LLM → Answer
```

this project uses:

```text
Question
   ↓
Retrieve relevant document information
   ↓
Context + Question
   ↓
LLM
   ↓
Answer
```

---

## 📈 Future Improvements

* Support multiple PDFs simultaneously
* Maintain separate vector stores for different documents
* Add chat history and conversational memory
* Display document sources and page numbers
* Add document preview
* Improve chunking strategies
* Add retrieval evaluation
* Add streaming responses
* Add support for additional document formats such as DOCX and TXT

---

## 👨‍💻 Author

**Yash Raj**

4th Year ECE Student at NMIT Bangalore

Interested in **AI/ML, LLMs, RAG and Full-Stack Development**.

---

## ⭐ Project Highlights

This project demonstrates practical implementation of:

* Retrieval-Augmented Generation
* Vector databases
* Semantic search
* Document embeddings
* LLM integration
* Prompt engineering
* Streamlit deployment
* LangChain-based AI pipelines

If you found this project useful, consider giving the repository a ⭐.
