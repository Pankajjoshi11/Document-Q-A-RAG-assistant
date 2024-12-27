import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama
from sentence_transformers import CrossEncoder
from streamlit.runtime.uploaded_file_manager import UploadedFile


system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


# Set page configuration at the top
st.set_page_config(page_title="RAG-based LLM Model")

def process_document(uploaded_file: UploadedFile) -> list[Document]:
    file_extension = os.path.splitext(uploaded_file.name)[-1]
    
    # Ensure the suffix is a string
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=file_extension, delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()  # Ensure data is written before processing

    try:
        # Load the file using PyMuPDFLoader
        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
    except Exception as e:
        raise RuntimeError(f"Error loading document: {e}")
    finally:
        # Delete the temporary file after loading
        os.unlink(temp_file.name)

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    return text_splitter.split_documents(docs)

def get_vector_collection() -> chromadb.Collection:
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(
        name="rag-app",
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(all_splits: list[Document], file_name: str):
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []

    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    st.success("Data added to the vector store successfully!")


def query_collection(prompt:str, n_results:int =10):
    collection = get_vector_collection()
    results=collection.query(query_texts=[prompt],n_results=n_results)
    return results


def call_llm(context:str,prompt:str):
    response=ollama.chat(
        model='llama3.2:3b',
        stream=True,
        messages=[
            {
                "role":'system',
                "content":system_prompt
            },
            {
                "role":"user",
                "content":f"Context:{context},Question:{prompt}"
            }
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(documents: list[str], prompt: str) -> tuple[str, list[int]]:
    relevant_text = ""
    relevent_text_ids = []
    encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Note: it's "marco" not "macro"
    ranks = encoder_model.rank(prompt, documents, top_k=3)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevent_text_ids.append(rank["corpus_id"])
    return relevant_text, relevent_text_ids


if __name__ == "__main__":
    st.sidebar.header("RAG Question Answer")
    uploaded_file = st.sidebar.file_uploader(
        label="Upload a file",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )
    process = st.sidebar.button("Process")

    if uploaded_file and process:
        try:
            # Normalize filename for consistent IDs
            normalized_file_name = uploaded_file.name.translate(
                str.maketrans({"-": "_", ".": "_", " ": "_"})
            )
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, normalized_file_name)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    st.header('RAG document summarizer')
    prompt=st.text_area("Ask a question related to your document or summarize it")
    ask=st.button("Ask")

    if ask and prompt:
        results=query_collection(prompt)
        context=results.get('documents')[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(context,prompt)
        response = call_llm(context=relevant_text, prompt=prompt)        
        st.write_stream(response)

        with st.expander("See retrieved documents"):
                st.write(results)

        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)
            st.write(relevant_text)