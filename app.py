import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ChatVectorDBChain
from langchain.chat_models import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
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
# Файлды уақытша сақтау
with open("temp.pdf", "wb") as f:
    f.write(uploaded_file.getbuffer())

# PDF мәтінін оқу
with open("temp.pdf", "rb") as f:
    reader = PdfReader(f)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

# Уақытша файлды жою
os.remove("temp.pdf")

# Мәтінді бөлу
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.create_documents([raw_text])
texts = [doc.page_content for doc in documents]

# Embedding объектісін құру
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_embeddings_with_retry(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings.embed_documents(texts)

# FAISS вектор қорын құру
try:
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts, embeddings)
    st.success("PDF құжат өңделіп, вектор қоры дайындалды!")
except Exception as e:
    st.error(f"FAISS қате: {e}")
    st.stop()

# Чат функциясы
chat_model = ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo")
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
