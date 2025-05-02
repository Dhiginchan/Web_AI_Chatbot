from flask import Flask, render_template, request, jsonify
import os
import dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain

# Load .env variables
dotenv.load_dotenv()

# Get API key and model from environment
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing!")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0.7, google_api_key=GOOGLE_API_KEY)

# Create custom prompt
TEMPLATE = """
You are an advanced AI assistant designed to provide **intelligent, structured, and well-explained answers** to users. Your goal is to help users with different types of queries.

📝 **Past Conversations**:
{history}

💬 **User's Current Query**:
{input}

---

## **🛠 How to Answer Based on Use Case**

### **1️⃣ Chatbots (General Conversations)**
- Engage in natural, friendly, and human-like conversations.
- Keep the discussion interactive by asking relevant follow-up questions.

### **2️⃣ Customer Support**
- Provide **step-by-step guidance** for user queries.
- If needed, direct users to external resources for further help.

### **3️⃣ Education & Tutoring**
- Break down **complex concepts into simple explanations**.
- Offer **real-world examples** to improve understanding.

### **4️⃣ Healthcare Assistance**
- Give **general wellness advice** but **avoid medical diagnoses**.
- Always recommend consulting a healthcare professional for serious concerns.

📌 **General Guidelines**:
- Be **accurate, structured, and engaging** in responses.
- If a question is beyond your knowledge, say: **"I don’t have enough data on this, but here’s what I do know..."**
- Adapt the tone based on the use case (friendly, professional, or educational).
"""


prompt = PromptTemplate.from_template(TEMPLATE)

# Use summary memory
memory = ConversationSummaryMemory(memory_key="history", llm=llm)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt)

# Initialize Flask app
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
    # Get port from environment or default to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
