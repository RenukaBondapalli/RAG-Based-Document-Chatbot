from flask import Flask, request, render_template_string, redirect, url_for
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import chromadb
import shutil
import gc
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY is not set in environment variables")
os.environ["GOOGLE_API_KEY"] = api_key

# Flask app setup
app = Flask(__name__)
app.secret_key = "123"
UPLOAD_FOLDER = "uploads"
CHROMA_DB_DIR = "./chroma_db"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load embedding model
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

# In-memory chat history
chat_history = []

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Chatbot</title>
    <style>
        body { font-family: Arial; background-color: #f5f5f5; padding: 30px; }
        .chat-box { background: #fff; padding: 20px; border-radius: 8px; max-width: 700px; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        .message { margin-bottom: 15px; }
        .user { color: #007bff; }
        .bot { color: #28a745; }
        .upload-form { margin-bottom: 20px; }
        input[type="file"], input[type="text"], input[type="submit"] {
            padding: 10px; border-radius: 4px; border: 1px solid #ccc; width: 100%;
        }
        input[type="submit"] { background-color: #007bff; color: white; cursor: pointer; }
    </style>
</head>
<body>
    <div class="chat-box">
        <h2>RAG Chatbot</h2>
        <form method="POST" enctype="multipart/form-data" class="upload-form">
            <input type="file" name="file" accept=".txt,.pdf"><br><br>
            <input type="text" name="question" placeholder="Ask a question..." required><br><br>
            <input type="submit" value="Ask">
        </form>
        <form method="POST" action="/clear" style="margin-top: 10px;">
            <input type="submit" value="Clear Chat" style="background-color: #dc3545; color: white;">
        </form>
        <div class="messages">
            {% for q, a in chat_history %}
                <div class="message"><strong class="user">You:</strong> {{ q }}</div>
                <div class="message"><strong class="bot">Bot:</strong> {{ a }}</div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""

# Function to safely clear Chroma DB
def clear_chroma_db(path):
    try:
        client = chromadb.PersistentClient(path=path)
        client.reset()
    except Exception as e:
        print("Chroma reset error:", e)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    gc.collect()

@app.route("/", methods=["GET", "POST"])
def index():
    global chat_history
    persist_dir = CHROMA_DB_DIR

    if request.method == "POST":
        file = request.files.get("file")
        question = request.form.get("question")

        if file and file.filename:
            clear_chroma_db(persist_dir)

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            loader = PyPDFLoader(filepath) if filename.lower().endswith(".pdf") else TextLoader(filepath)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = splitter.split_documents(documents)

            vectordb = Chroma.from_documents(docs, embedding, persist_directory=persist_dir)
            vectordb.persist()
        else:
            if not os.path.exists(persist_dir):
                return render_template_string(HTML_TEMPLATE, chat_history=chat_history)
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
        answer = qa.run(question)

        chat_history.append((question, answer))

    return render_template_string(HTML_TEMPLATE, chat_history=chat_history)

@app.route("/clear", methods=["POST"])
def clear_chat():
    global chat_history
    chat_history = []
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)