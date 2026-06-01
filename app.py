import streamlit as st
import numpy as np
import faiss
import torch
import torch.nn.functional as F
import requests
import re
import fitz
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

def bm25_preprocess(text):
    return re.sub(r'[^\w\s]', '', text).lower().split()

def mean_pooling(model_op, attention_mask):
    token_embeds = model_op[0]
    ip_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    sum_embeddings = torch.sum(token_embeds * ip_mask_expanded, 1)
    sum_mask = torch.clamp(ip_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

st.set_page_config(page_title="Project Brain", layout="wide")

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    return tokenizer, model, reranker, device

tokenizer, model, reranker, device = load_models()

#session state (like django)
if "index" not in st.session_state:
    with st.spinner("Loading 35,000+ chunks..."):
        data = np.load('rag_foundation.npz', allow_pickle=True)
        st.session_state.knowledge_chunk = data['metadata']
        kb_embeddings = data['embeddings'].astype('float32')
        
        dim = kb_embeddings.shape[1]
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(kb_embeddings)
        
        st.session_state.corpus = [bm25_preprocess(chunk['embedding_text']) for chunk in st.session_state.knowledge_chunk]
        st.session_state.bm25 = BM25Okapi(st.session_state.corpus)
        st.session_state.messages = []
        st.success("Brain Foundation Loaded!")

def get_query_embedding(text):
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors='pt', max_length=256).to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy().astype('float32')

def retrieve_context(question, k_get=100, k_f=5):
    #vector
    q_embed = get_query_embedding(question)
    _, vdb_indices = st.session_state.index.search(q_embed, k_get)
    
    #bm25
    tok_query = bm25_preprocess(question)
    bm25_scores = st.session_state.bm25.get_scores(tok_query)
    bm25_global_top = np.argsort(bm25_scores)[::-1][:k_get]
    
    #hybrid
    hybrid_indices = list(set(vdb_indices[0]) | set(bm25_global_top))
    candidate_chunks = [st.session_state.knowledge_chunk[idx] for idx in hybrid_indices]
    
    #reranker
    pairs = [[question, c['embedding_text']] for c in candidate_chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk['text'] for chunk, score in ranked[:k_f]]

def call_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    payload = {"model": "llama3", "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    try:
        response = requests.post(url, json=payload)
        return response.json().get('response', "Error: No response.")
    except Exception as e:
        return f"Ollama Connection Error: {e}"

#ingestion
def handle_pdf_upload(uploaded_file):
    
    with st.spinner("Processing PDF..."):
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50, separators=["\n\n", "\n", " ", ""])
        new_raw_chunks = splitter.split_text(text)
        
        new_formatted = []
        new_embeddings = []
        for chunk in new_raw_chunks:
            new_formatted.append({'embedding_text': chunk, 'text': chunk, 'metadata': {'domain': 'upload'}})
            new_embeddings.append(get_query_embedding(chunk))
        
        #faiss update
        new_embeddings_final = np.vstack(new_embeddings).astype('float32')
        st.session_state.index.add(new_embeddings_final)
        
        #metadata + bm25 update
        st.session_state.knowledge_chunk = np.append(st.session_state.knowledge_chunk, new_formatted)
        new_corpus_entries = [bm25_preprocess(c) for c in new_raw_chunks]
        st.session_state.corpus.extend(new_corpus_entries)
        st.session_state.bm25 = BM25Okapi(st.session_state.corpus)
        st.success(f"Added {len(new_raw_chunks)} chunks to memory!")

with st.sidebar:
    st.title("Brain Settings")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Ingest PDF"):
        handle_pdf_upload(uploaded_file)

st.title("Hybrid RAG")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("Thinking..."):

        context_chunks = retrieve_context(prompt)
        
        if not context_chunks:
            answer = "I couldn't find any relevant documents to answer that question."
        else:
            context_str = "\n\n".join([f"Source {i+1}\n{text}" for i, text in enumerate(context_chunks)])

            final_prompt = f"""
            You are a helpful assistant. 
            Answer the user's question ONLY using the provided context. 

            STRICT RULES:
            1. Use ONLY the provided context. If the answer isn't there, say you don't know.
            2. Be concise. Start with the most important information.
            3. Prioritize specific timelines (e.g., "within 10 days"), legal requirements, or form names. Remove sources from the answer.
            4. Do not offer general life advice or information not found in the snippets.

            CONTEXT:
            {context_str}

            USER QUESTION:
            {prompt}

            HELPFUL ANSWER:
            """
            
            answer = call_ollama(final_prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)