import streamlit as st
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

# Streamlit Config
st.set_page_config(page_title="News Research Tool", layout="wide")

# Session State

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

# LLM Model

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.6,
    max_tokens=500,
    google_api_key=st.secrets['GEMINI_API_KEY']
)

# CACHED EMBEDDINGS
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return SentenceTransformerEmbeddings(
        model_name="all-mpnet-base-v2"
    )

# CACHED VECTOR STORE

@st.cache_resource(show_spinner=True)
def build_vectorstore(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=500,
        chunk_overlap=200
    )

    docs = splitter.split_documents(data)
    embeddings = load_embeddings()

    return FAISS.from_documents(docs, embeddings)

# UI

st.title("📰 News Articles Research Tool")
st.sidebar.title("Paste Article URLs")

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_button = st.sidebar.button("Process URLs")

# Build Vector Store (ONCE)

if process_button and urls:
    with st.spinner("Processing articles..."):
        st.session_state.vectorstore = build_vectorstore(tuple(urls))
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )
    st.success("Vector store ready ✅")

# Question Answering
query = st.text_input("Ask a question about the articles:")

if query:
    if st.session_state.retriever is None:
        st.warning("Please process URLs first.")
    else:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=st.session_state.retriever
        )

        with st.spinner("Generating answer..."):
            result = chain({"question": query}, return_only_outputs=True)

        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("sources"):
            st.subheader("Sources")
            for src in result["sources"].split("\n"):
                st.write(src)
