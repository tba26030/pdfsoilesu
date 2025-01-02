import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
import openai

# Бағдарлама атауы
st.set_page_config(page_title="PDF құжатпен қазақша сөйлесу", layout="wide")

# Бағдарлама шапкасы
st.title("PDF құжатпен қазақша сөйлесу")
st.markdown(
    "Бұл бағдарлама PDF құжаттардан қазақша сұрақтарға жауап береді. OpenAI API пайдаланылады. "
    "PDF құжаттағы ақпаратқа сәйкес ықтимал сұрақтардың тізімін де ұсынады."
)

# OpenAI API кілтін енгізу
api_key = st.text_input("OpenAI API кілтіңізді енгізіңіз:", type="password")
if not api_key:
    st.warning("Жалғастыру үшін OpenAI API кілтін енгізіңіз.")
    st.stop()

openai.api_key = api_key

# PDF файл жүктеу
uploaded_file = st.file_uploader("PDF файлды жүктеңіз:", type="pdf")
if not uploaded_file:
    st.info("PDF құжатты жүктеген соң бағдарламаны пайдалана аласыз.")
    st.stop()

# PDF құжатты өңдеу
st.info("PDF құжат өңделуде, күтіңіз...")
loader = PyPDFLoader(uploaded_file)
documents = loader.load()

# Embedding және вектор қоры
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)
st.success("PDF құжат өңделіп, вектор қоры дайындалды!")

# Чат функциясы
chat_model = ChatOpenAI(temperature=0.5, model_name="gpt-4o")
qa_chain = ChatVectorDBChain.from_llm(chat_model, vectorstore)

# Сұрақ енгізу
st.subheader("Сұрақ қойыңыз")
user_query = st.text_input("Сіздің сұрағыңыз:")
if user_query:
    with st.spinner("Жауап ізделуде..."):
        response = qa_chain.run(user_query)
        st.markdown(f"**Жауап:** {response}")

        # Ықтимал сұрақтарды ұсыну
        follow_up_questions = [
            "Бұл тақырыпқа байланысты тағы не білуге болады?",
            "Құжаттағы негізгі фактілер қандай?",
            "Мәтіндегі ұқсас тақырыптар туралы айтып беріңіз."
        ]
        st.subheader("Ықтимал сұрақтар")
        for i, question in enumerate(follow_up_questions, 1):
            st.write(f"{i}. {question}")

# Қосымша ақпарат
st.markdown(
    "---\nБұл бағдарламаны жасаған [Тимур Бектұр](https://www.instagram.com/timurbektur). "
    "Егер сіз де жасанды интеллект арқылы бағдарлама жасауды үйренгіңіз келсе немесе басқа да идеялар бойынша бағдарламалар жасағыңыз келсе, маған хабарласыңыз."
)
