"""Task 9: multi-turn test of the RAG chatbot (headless, via Streamlit AppTest)."""
from streamlit.testing.v1 import AppTest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = open("data/sample.txt", encoding="utf-8").read()
chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(
    [Document(page_content=text, metadata={"source": "sample.txt"})]
)
retriever = FAISS.from_documents(
    chunks, HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
).as_retriever(search_kwargs={"k": 3})

at = AppTest.from_file("app.py", default_timeout=180)
at.session_state["retriever"] = retriever  # file_uploader can't be simulated
at.session_state["index_signature"] = ()
at.run()

turns = [
    ("1. Initial factual question", "What are the two stages of a RAG pipeline?"),
    ("2. Follow-up question", "What happens in the second one?"),
    ("3. Out-of-context question", "Who won the 2018 FIFA World Cup?"),
]
for label, q in turns:
    at.chat_input[0].set_value(q).run()
    assert not at.exception, at.exception
    m = at.session_state["messages"][-1]
    print(f"{label}\nQ: {q}\nA: {m['content']}\nsources: {[s['text'][:40] for s in m['sources']]}\n")

msgs = at.session_state["messages"]
print("History length:", len(msgs), "messages;", [m["role"] for m in msgs])
assert len(msgs) == 6, "chat history not maintained"
assert "don't know" in msgs[-1]["content"].lower(), "out-of-context answer not refused"
print("PASS")
