from flask import Flask, render_template, request, jsonify
import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

dotenv.load_dotenv()

# Load API credentials
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing!")

# Initialize LLM and prompt
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)

TEMPLATE = """
You are an advanced AI assistant designed to provide **intelligent, structured, and well-explained answers** to users.

📝 Past Conversations:
{history}

💬 User's Current Query:
{input}
"""

prompt = PromptTemplate.from_template(TEMPLATE)
memory = ConversationSummaryMemory(memory_key="history", llm=llm)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

# Flask app setup
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    result = conversation.invoke({"input": user_input})
    return jsonify({"response": result["response"]})

if __name__ == "__main__":
    app.run(debug=True)
