import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os
import openai
from openai import OpenAI
import time

# Streamlit page configuration
st.set_page_config(
    page_title="PDF құжатпен қазақша сөйлесу",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .css-1d391kg {
        padding: 2rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .error {
        color: red;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📚 PDF құжатпен қазақша сөйлесу")
st.markdown("PDF құжаттарыңызбен қазақ тілінде сөйлесіңіз. Кез-келген тілдегі PDF құжаттарға қазақша сұрақ қоя аласыз.")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'error' not in st.session_state:
    st.session_state.error = None

# API key input
api_key = st.text_input("OpenAI API кілтін енгізіңіз:", type="password")

def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def process_pdf(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Load and split the PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Create conversation chain
        llm = ChatOpenAI(
            temperature=0.7,
            model_name='gpt-3.5-turbo',  # Changed to gpt-3.5-turbo for better stability
            openai_api_key=api_key
        )
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
        )

        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return conversation_chain
    except Exception as e:
        st.session_state.error = f"Қате орын алды: {str(e)}"
        return None

def generate_suggested_questions(context):
    try:
        if not api_key:
            return []
        
        llm = ChatOpenAI(
            temperature=0.7, 
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo'
        )
        prompt = f"""
        Берілген контекст негізінде 3 ықтимал сұрақ ұсыныңыз. 
        Сұрақтар қазақ тілінде болуы керек.
        Контекст: {context}
        """
        response = llm.predict(prompt)
        questions = [q.strip() for q in response.split('\n') if q.strip()]
        return questions[:3]
    except Exception:
        return []

# Validate API key when provided
if api_key:
    if not validate_api_key(api_key):
        st.error("Жарамсыз API кілті. Өтінеміз, тексеріп қайта енгізіңіз.")
        st.stop()

# File upload
uploaded_file = st.file_uploader("PDF құжатты жүктеңіз", type="pdf")

if uploaded_file and api_key:
    if not st.session_state.conversation:
        with st.spinner("Құжат өңделуде..."):
            st.session_state.conversation = process_pdf(uploaded_file)
            if st.session_state.conversation:
                st.success("Құжат сәтті жүктелді!")

# Show error if exists
if st.session_state.error:
    st.error(st.session_state.error)
    st.session_state.error = None

# Chat interface
if st.session_state.conversation:
    # Chat input
    user_question = st.text_input("Сұрағыңызды қазақ тілінде жазыңыз:")
    
    # Suggested questions buttons
    if st.session_state.suggested_questions:
        st.write("Ықтимал сұрақтар:")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, question in enumerate(st.session_state.suggested_questions):
            if cols[i].button(question, key=f"suggested_q_{i}"):
                user_question = question

    if user_question:
        try:
            with st.spinner("Жауап іздеуде..."):
                response = st.session_state.conversation({
                    'question': f"""
                    Сұраққа қазақ тілінде жауап беріңіз. 
                    Егер құжатта жауап табылмаса, оны айтыңыз.
                    Сұрақ: {user_question}
                    """
                })
                
                # Store chat history
                st.session_state.chat_history.append((user_question, response['answer']))
                
                # Generate new suggested questions
                st.session_state.suggested_questions = generate_suggested_questions(response['answer'])

                # Refresh the page to show new content
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Қате орын алды: {str(e)}")

# Display chat history
for question, answer in st.session_state.chat_history:
    st.write(f"🙋‍♂️ **Сұрақ:** {question}")
    st.write(f"🤖 **Жауап:** {answer}")
    st.markdown("---")

# Footer
st.markdown("""
---
Бұл бағдарламаны жасаған Тимур Бектұр. 
Егер сіз де жасанды интеллект арқылы бағдарлама жасауды үйренгіңіз келсе, 
өзге де идеялар бойынша ЖИ арқылы өзіңізге қосымша жасағыңыз келсе, 
маған хабарласыңыз: instagram @timurbektur
""")
