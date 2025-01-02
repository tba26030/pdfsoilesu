import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
import tempfile
import os
from openai import OpenAI
from typing import List, Tuple

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
    .stButton>button {
        width: 100%;
    }
    .error {
        color: red;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .user-message {
        background-color: #f8fafc;
    }
    .bot-message {
        background-color: #f0f9ff;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📚 PDF құжатпен қазақша сөйлесу")
st.markdown("PDF құжаттарыңызбен қазақ тілінде сөйлесіңіз. Кез-келген тілдегі PDF құжаттарға қазақша сұрақ қоя аласыз.")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'suggested_questions' not in st.session_state:
    st.session_state.suggested_questions = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

# API key input
api_key = st.text_input("OpenAI API кілтін енгізіңіз:", type="password")

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key"""
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def process_pdf(uploaded_file) -> FAISS:
    """Process PDF file and create vector store"""
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

        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return vectorstore
    except Exception as e:
        st.error(f"Құжатты өңдеу кезінде қате орын алды: {str(e)}")
        return None

def create_chain(vectorstore: FAISS):
    """Create conversation chain"""
    # Initialize LLM
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4",
        openai_api_key=api_key
    )

    # Create the retrieval chain
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Create the prompt template
    template = """
    Сен PDF құжаттың мазмұнын талдай алатын көмекшісің.
    Құжаттың контексті:
    {context}
    
    Чат тарихы:
    {chat_history}
    
    Адам: {question}
    Көмекші: Қазақ тілінде жауап беремін:
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, 
         "chat_history": RunnablePassthrough(), 
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def format_chat_history(history: List[Tuple[str, str]]) -> str:
    """Format chat history for the prompt"""
    formatted = []
    for human, ai in history:
        formatted.append(f"Адам: {human}")
        formatted.append(f"Көмекші: {ai}")
    return "\n".join(formatted)

def generate_suggested_questions(context: str) -> List[str]:
    """Generate suggested questions based on context"""
    try:
        if not api_key:
            return []
        
        llm = ChatOpenAI(
            temperature=0.7, 
            openai_api_key=api_key,
            model="gpt-4"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Сен қазақ тілінде сұрақтар ұсынатын көмекшісің."),
            ("user", "Берілген контекст бойынша 3 мағыналы сұрақ ұсын:\n\n{context}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": context})
        
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
    if st.session_state.vectorstore is None:
        with st.spinner("Құжат өңделуде..."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
            if st.session_state.vectorstore:
                st.success("Құжат сәтті жүктелді!")

# Chat interface
if st.session_state.vectorstore:
    # Create chain
    chain = create_chain(st.session_state.vectorstore)
    
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
                # Format chat history
                chat_history = format_chat_history(st.session_state.chat_history)
                
                # Get response
                response = chain.invoke({
                    "question": user_question,
                    "chat_history": chat_history
                })
                
                # Store chat history
                st.session_state.chat_history.append((user_question, response))
                
                # Generate new suggested questions
                st.session_state.suggested_questions = generate_suggested_questions(response)

                # Refresh the page to show new content
                st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Жауап алу кезінде қате орын алды: {str(e)}")

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    # User message
    st.markdown(f"""
    <div class="chat-message user-message">
        <b>🙋‍♂️ Сұрақ:</b><br>{question}
    </div>
    """, unsafe_allow_html=True)
    
    # Bot message
    st.markdown(f"""
    <div class="chat-message bot-message">
        <b>🤖 Жауап:</b><br>{answer}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
---
Бұл бағдарламаны жасаған Тимур Бектұр. 
Егер сіз де жасанды интеллект арқылы бағдарлама жасауды үйренгіңіз келсе, 
өзге де идеялар бойынша ЖИ арқылы өзіңізге қосымша жасағыңыз келсе, 
маған хабарласыңыз: instagram @timurbektur
""")
