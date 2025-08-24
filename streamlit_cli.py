import streamlit as st
import time
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from src.rag_ft_qa import generate_response, retrieve_documents


@st.cache_resource
def load_fine_tuned_generator():
    tokenizer = AutoTokenizer.from_pretrained("./sft_model")
    model = AutoModelForCausalLM.from_pretrained("./sft_model")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
llm = pipeline("text-generation", model="gpt2", device=-1)
fine_tuned_generator = load_fine_tuned_generator()

st.title("RAG & Fine-Tuned Q&A System")

mode = st.radio("Select Mode:", ["RAG", "Fine-Tuned"])
query = st.text_input("Enter your question:")

if st.button("Get Answer") and query:
    if mode == "RAG":
        st.info("Using RAG model to generate the response...")
        start_time = time.time()
        top_docs = retrieve_documents(query)
        response = generate_response(llm, query, top_docs, start_time)
    else:
        st.info("Using Fine-Tuned model to generate the response...")
        start_time = time.time()
        response = generate_response(fine_tuned_generator, query, "", start_time)

    st.subheader("Answer:")
    st.write(response)