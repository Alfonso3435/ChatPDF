import streamlit as st
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI  # Usamos ChatOpenAI para modelos de chat
from dotenv import load_dotenv
import os

# Cargar las variables de entorno
load_dotenv()

# Obtener la clave de OpenAI desde .env
def getOpenAIkey():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    return OPENAI_API_KEY

api_key = getOpenAIkey()

# Verificar el dispositivo disponible (CUDA o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el PDF
loader = PyPDFLoader(file_path=r".\monopoly.pdf")
data = loader.load()

# Dividir el texto en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(data)

# Crear embeddings para los fragmentos de texto
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_documents(text_chunks, embeddings)

# Crear memoria para el historial de conversación
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Crear la cadena conversacional para recuperación
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4", temperature=0.75, openai_api_key=api_key),
    retriever=db.as_retriever(),
    memory=memory
)

# Crear la aplicación en Streamlit
st.title("PDF Chatbot con OpenAI y Langchain")

# Estado para mantener el historial de la conversación
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Entrada del usuario
user_input = st.text_input("Haz una pregunta:")

# Si el usuario ingresa algo, procesamos la pregunta
if user_input:
    # Obtener respuesta del LLM
    result = qa_chain({"question": user_input})
    answer = result["answer"]
    
    # Almacenar la pregunta y respuesta en el historial de chat
    st.session_state["chat_history"].append((user_input, answer))

# Mostrar el historial de conversación
if st.session_state["chat_history"]:
    for i, (question, answer) in enumerate(st.session_state["chat_history"]):
        st.write(f"**Tú ({i+1}):** {question}")
        st.write(f"**Chatbot:** {answer}")

