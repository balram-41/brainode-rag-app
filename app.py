import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Page Configuration
st.set_page_config(page_title="BraiNode: Conversational RAG", page_icon="🤖", layout="wide")

st.title("📚 BraiNode: Conversational Academic RAG")
st.markdown("Upload PDFs and have a natural, multi-turn conversation with your documents!")

# 2. Sidebar Setup
st.sidebar.header("1. API Configuration")
api_key = st.sidebar.text_input("Enter Google Gemini API Key:", type="password")

st.sidebar.header("2. Document Management")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF Documents", 
    type=["pdf"], 
    accept_multiple_files=True
)

process_btn = st.sidebar.button("⚡ Process & Vectorize Documents")
clear_btn = st.sidebar.button("🗑️ Clear Chat History")

# Initialize Chat Memory in Streamlit Session State
if "chat_history" not in st.session_state or clear_btn:
    st.session_state.chat_history = []
    
if "langchain_history" not in st.session_state or clear_btn:
    st.session_state.langchain_history = []

if "rag_chain" not in st.session_state or clear_btn:
    st.session_state.rag_chain = None

# 3. Vectorization & Pipeline Processing
if process_btn:
    if not api_key:
        st.sidebar.error("Please enter your Gemini API key first!")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one PDF file!")
    else:
        os.environ["GOOGLE_API_KEY"] = api_key
        with st.spinner("Processing documents into vector database..."):
            documents = []
            
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                documents.extend(loader.load())
                os.remove(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = QdrantVectorStore.from_documents(
                chunks,
                embedding=embeddings,
                location=":memory:",
                collection_name="conversational_docs"
            )

            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)

            # --- NEW: History-Aware Retriever Prompt ---
            contextualize_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
            contextualize_q_prompt = ChatPromptTemplate.from_messages([
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(
                llm, retriever, contextualize_q_prompt
            )

            # --- NEW: Question-Answering Prompt with Memory ---
            qa_system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know.\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
            st.session_state.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
            
            st.sidebar.success(f"Successfully processed {len(chunks)} text chunks! Ready to chat.")

# 4. Interactive Chat Interface
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_query := st.chat_input("Ask a question about your uploaded documents..."):
    if not st.session_state.rag_chain:
        st.error("⚠️ Please upload and process at least one PDF document using the sidebar first!")
    else:
        # Save and display user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate answer using conversational chain
        with st.chat_message("assistant"):
            with st.spinner("Searching document context and chat history..."):
                response = st.session_state.rag_chain.invoke({
                    "input": user_query,
                    "chat_history": st.session_state.langchain_history
                })
                
                answer = response["answer"]
                st.markdown(answer)

                with st.expander("View Source Citations"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')})**")
                        st.text(doc.page_content[:300] + "...")

        # Update both Streamlit UI history and LangChain logical history
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.langchain_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=answer)
        ])