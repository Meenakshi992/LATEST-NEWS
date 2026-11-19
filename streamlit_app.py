import re
import sys
import os

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import streamlit as st
from openai import OpenAI
import fitz
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_DB_PATH = os.path.abspath("./chroma_db")
DEFAULT_COLLECTION = "my_documents"
COLLECTION_METADATA = {"hnsw:space": "cosine"}

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=DEFAULT_DB_PATH)


# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management

openai_api_key = st.secrets.get("OPENAI_API_KEY")
# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Step 1: Read and Extract document
def step1_extract_text_from_file(uploaded_file) -> str:
    uploaded_file.seek(0)
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)
    # Unindented so .txt files actually reach this line
    return uploaded_file.read().decode("utf-8", errors="ignore")

# Step 2: Clean text
def step2_clean_text(text: str, remove_urls: bool, normalize_space: bool, lowercase: bool) -> str:
    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", "", text)
    if normalize_space:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
    if lowercase:
        text = text.lower()
    return text.strip()

# Step 3: Chunking
def step3_chunk_text(cleaned_text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(cleaned_text)

# Step 4: Embedding
def step4_get_openai_embeddings(texts: list[str], api_key: str, dimensions: int) -> list[list[float]]:
    client = OpenAI(api_key=api_key)
    sanitized_texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(
        input=sanitized_texts,
        model="text-embedding-3-small",
        dimensions=dimensions
    )
    return [item.embedding for item in response.data]

# Step 5: Save to ChromaDB
def step5_save_to_chromadb(
    chunks: list[str],
    vectors: list[list[float]],
    source_name: str,
    collection_name: str = DEFAULT_COLLECTION,
) -> int:
    if not chunks or not vectors:
        return 0

    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=COLLECTION_METADATA
    )

    ids = [f"{source_name}_chunk_{idx}" for idx in range(len(chunks))]
    metadatas = [{"source": source_name, "chunk_id": idx} for idx in range(len(chunks))]

    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=metadatas
    )
    return collection.count()

# --- UI Layout ---
st.title("📄 Document Cleaning, Chunking & OpenAI Embeddings")

with st.sidebar:
    st.header("1. Set Embedding Dimensions")
    dimensions = st.selectbox(
        "Embedding Dimensions",
        options=[1536, 512, 256],
        index=0
    )

    st.header("2. Cleaning Options")
    opt_whitespace = st.checkbox("Normalize extra whitespace", value=True)
    opt_urls = st.checkbox("Remove URLs", value=False)
    opt_lowercase = st.checkbox("Convert to lowercase", value=False)

    st.header("3. Chunking Settings")
    chunk_size = st.slider("Chunk Size (characters)", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=500, value=50, step=10)

uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded_file is not None and st.button("Process, Embed & Save", type="primary"):
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        try:
            # 1. Extract & Clean
            rawtext = step1_extract_text_from_file(uploaded_file)
            cleaned_text = step2_clean_text(rawtext, opt_urls, opt_whitespace, opt_lowercase)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Raw Text Preview")
                st.text_area("Raw", rawtext[:1500], height=200, disabled=True)
            with col2:
                st.subheader("Cleaned Text Preview")
                st.text_area("Cleaned", cleaned_text[:1500], height=200, disabled=True)

            # 2. Chunking
            chunks = step3_chunk_text(cleaned_text, chunk_size, chunk_overlap)
            st.info(f"Generated **{len(chunks)}** chunks from **{len(cleaned_text):,}** characters.")

            # 3. Embeddings (using sidebar `dimensions`, not 1534)
            with st.spinner(f"Generating embeddings ({dimensions}d) for {len(chunks)} chunks..."):
                vectors = step4_get_openai_embeddings(chunks, openai_api_key, dimensions)

            # 4. Save to ChromaDB AFTER chunks and vectors are created
            total_in_db = step5_save_to_chromadb(
                chunks=chunks,
                vectors=vectors,
                source_name=uploaded_file.name
            )
            st.success(f"Saved! Collection `{DEFAULT_COLLECTION}` now has **{total_in_db}** total chunks.")

            # 5. Preview Table
            df_preview = pd.DataFrame([
                {
                    "Chunk ID": idx,
                    "Text Snippet": chunk[:100] + "...",
                    "Dimensions": len(vec)
                }
                for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
            ])
            st.dataframe(df_preview, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")


# --- Query Function looking your searched documents ---

def query_chromadb(
    query_text: str,
    api_key: str,
    dimensions: int,
    collection_name: str = DEFAULT_COLLECTION,
    top_k: int = 3
) -> dict:
    """Embeds the user's question and retrieves top matching chunks."""
    query_vector = step4_get_openai_embeddings([query_text], api_key, dimensions)
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=COLLECTION_METADATA
    )

    return collection.query(
        query_embeddings=query_vector,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

# --- Search & Q&A UI ---
st.divider()
st.header("Ask Questions About Uploaded Document")

client = get_chroma_client()
collection = client.get_or_create_collection(
    name=DEFAULT_COLLECTION,
    metadata=COLLECTION_METADATA
)

total_chunks = collection.count()

if total_chunks == 0:
    st.info("No chunks found in ChromaDB yet. Upload and process a document above first.")
else:
    st.caption(f"Connected to `{DEFAULT_COLLECTION}` ({total_chunks} chunks indexed).")

    col_q, col_k = st.columns([4, 1])
    with col_q:
        user_question = st.text_input("Enter your question:", placeholder="tell me about this document?")
    with col_k:
        top_k = st.number_input("Top chunks", min_value=1, max_value=10, value=3)

    if st.button("Search & Answer", type="primary"):
        if not openai_api_key:
            st.error("Please add your OpenAI API key in `.streamlit/secrets.toml` or the sidebar.")
        elif not user_question.strip():
            st.warning("Please enter a question first.")
        else:
            with st.spinner("Searching ChromaDB and generating answer..."):
                # 1. Retrieve relevant chunks (make sure dimensions match what you saved!)
                results = query_chromadb(
                    query_text=user_question,
                    api_key=openai_api_key,
                    dimensions=dimensions,
                    top_k=int(top_k)
                )

                retrieved_docs = results["documents"][0]
                retrieved_meta = results["metadatas"][0]
                retrieved_dist = results["distances"][0]

                # 2. Build context prompt for OpenAI
                context_block = "\n\n---\n\n".join(
                    f"[Source: {meta['source']}, Chunk {meta['chunk_id']}]\n{doc}"
                    for doc, meta in zip(retrieved_docs, retrieved_meta)
                )

                ai_client = OpenAI(api_key=openai_api_key)
                chat_response = ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer the user's question using ONLY the provided context. If the answer is not in the context, say you do not have enough information."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context_block}\n\nQuestion: {user_question}"
                        }
                    ],
                    temperature=0.2
                )

                # 3. Display Answer & Sources
                st.subheader("Answer")
                st.write(chat_response.choices[0].message.content)

                with st.expander("View Retrieved Source Chunks"):
                    for doc, meta, dist in zip(retrieved_docs, retrieved_meta, retrieved_dist):
                        similarity = round((1 - dist) * 100, 1)
                        st.markdown(f"**{meta['source']}** (Chunk {meta['chunk_id']} — Similarity: `{similarity}%`)")
                        st.write(doc)
                        st.divider()
