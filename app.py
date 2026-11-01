import os
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the context below. "
     "If the answer is not in the context, say \"I don't know based on the provided context.\"\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

st.set_page_config(page_title="RAG chat", page_icon=":material/chat:")


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def get_chain():
    llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return RAG_PROMPT | llm | StrOutputParser()


def load_uploaded_file(uploaded):
    """Write the upload to a temp file and load it with the matching LangChain loader."""
    suffix = os.path.splitext(uploaded.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name
    try:
        loader = PyPDFLoader(path) if suffix == ".pdf" else TextLoader(path, encoding="utf-8")
        docs = loader.load()
    finally:
        os.remove(path)
    for d in docs:
        d.metadata["source"] = uploaded.name
    return docs


def build_retriever(files):
    docs = [d for f in files for d in load_uploaded_file(f)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text could be extracted (is the PDF a scanned image?).")
    store = FAISS.from_documents(chunks, get_embeddings())
    return store.as_retriever(search_kwargs={"k": 3}), len(chunks)


HISTORY_TURNS = 3  # previous user/assistant exchanges passed to the model

st.session_state.setdefault("messages", [])
st.session_state.setdefault("retriever", None)
st.session_state.setdefault("index_signature", None)
st.session_state.setdefault("chunk_count", 0)

st.title("Chat with your documents")

if not os.getenv("GROQ_API_KEY"):
    try:  # Streamlit Cloud: key comes from Settings -> Secrets, not .env
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to a .env file and restart the app.")
    st.stop()

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload PDF or text files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Clear chat", icon=":material/delete:", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

# Rebuild the vector index only when the set of uploaded files changes
signature = tuple((f.name, f.size) for f in files)
if signature != st.session_state.index_signature:
    st.session_state.index_signature = signature
    if files:
        try:
            with st.spinner("Indexing documents..."):
                st.session_state.retriever, st.session_state.chunk_count = build_retriever(files)
            st.session_state.messages = []
        except Exception as e:
            st.session_state.retriever = None
            st.session_state.chunk_count = 0
            st.error(f"Could not read the uploaded documents: {e}", icon=":material/error:")
    else:
        st.session_state.retriever = None
        st.session_state.chunk_count = 0

if st.session_state.retriever is None and not files:
    st.info("Upload a PDF or text file in the sidebar to start chatting.", icon=":material/upload_file:")
elif st.session_state.retriever is not None:
    st.sidebar.success(
        f"{len(files)} file(s) indexed into {st.session_state.chunk_count} chunks.",
        icon=":material/check_circle:",
    )

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(s["source"])
                    st.text(s["text"])

def recent_history():
    """Earlier turns as LangChain messages so follow-up questions make sense."""
    msgs = st.session_state.messages[-2 * HISTORY_TURNS:]
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in msgs
    ]


question = st.chat_input(
    "Ask a question about your documents",
    disabled=st.session_state.retriever is None,
    submit_mode="disable",
)

if question:
    history = recent_history()
    # Follow-ups like "why is that?" retrieve poorly alone, so add the previous question
    previous = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    search_query = f"{previous[-1]} {question}" if previous else question

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        try:
            # Retrieve relevant chunks, then generate the answer with ChatGroq
            docs = st.session_state.retriever.invoke(search_query)
            context = "\n\n".join(d.page_content for d in docs)
            answer = st.write_stream(
                get_chain().stream({"context": context, "history": history, "question": question})
            )
        except Exception as e:
            # Drop the failed question so the history stays clean and the user can retry
            st.session_state.messages.pop()
            st.error(f"Sorry, something went wrong while answering: {e}", icon=":material/error:")
            st.stop()
        elapsed = time.perf_counter() - started

        sources = [{"source": d.metadata.get("source", "unknown"), "text": d.page_content} for d in docs]
        st.caption(f"Answered in {elapsed:.1f} s")
        with st.expander("Sources"):
            for s in sources:
                st.caption(s["source"])
                st.text(s["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
