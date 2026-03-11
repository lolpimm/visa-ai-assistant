import os
import json
import psycopg2
from flask import Flask, request, jsonify, render_template
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("LLM_API_KEY")
db_url = os.getenv("DATABASE_URL")

app = Flask(__name__)

# --- CONFIGURATION ---
client = genai.Client(api_key=api_key)
MODEL_NAME = 'gemini-1.5-flash' # Using 2.0 Flash for better reliability/speed

@app.route('/')
def home():
    return render_template('index.html')

# ---------------------------------------------------------
# DATABASE HELPER FUNCTIONS
# ---------------------------------------------------------
def get_db_connection():
    return psycopg2.connect(db_url)

def get_current_prompt():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT prompt_text FROM ai_prompts ORDER BY id DESC LIMIT 1;")
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result[0] if result else "You are a helpful Thai Visa assistant."
    except Exception as e:
        print(f"DB Error: {e}")
        return "You are a helpful Thai Visa assistant."

def save_new_prompt(new_prompt_text):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO ai_prompts (prompt_text) VALUES (%s);", (new_prompt_text,))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"DB Save Error: {e}")

# ---------------------------------------------------------
# ROBUST PARSING HELPER
# ---------------------------------------------------------
def parse_ai_response(response_text, key_to_find):
    """
    Cleans the AI response and attempts to extract a value from JSON.
    If JSON parsing fails, it returns the raw text as a fallback.
    """
    try:
        clean_text = response_text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_text)
        if isinstance(data, dict):
            return data.get(key_to_find, response_text)
        return response_text
    except Exception:
        # Fallback for when Gemini returns plain text instead of JSON
        return response_text

# ---------------------------------------------------------
# API ENDPOINT 1: GENERATE REPLY
# ---------------------------------------------------------
@app.route('/generate-reply', methods=['POST'])
def generate_reply():
    try:
        data = request.json
        client_sequence = data.get('clientSequence', '')
        chat_history = data.get('chatHistory', [])

        current_prompt = get_current_prompt()
        
        # Limit history to last 5 messages to avoid token bloat/quota issues
        formatted_history = ""
        for msg in chat_history[-5:]:
            formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
        
        full_prompt = f"{current_prompt}\n\n{formatted_history}\nLATEST CLIENT MESSAGE: {client_sequence}\n\nReturn your answer in JSON format: {{\"reply\": \"...\"}}"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        # Use the robust parser to prevent 'AttributeError'
        reply = parse_ai_response(response.text, 'reply')
        return jsonify({"aiReply": reply})

    except Exception as e:
        # This will force the error to show up in your Railway Deploy Logs
        print(f"!!! CRITICAL ERROR: {str(e)}") 
        
        error_msg = str(e)
        if "429" in error_msg:
            return jsonify({"aiReply": "Server is busy (Quota hit). Please try again in 30 seconds!"}), 429
        return jsonify({"aiReply": f"Error: {error_msg}"}), 500

# ---------------------------------------------------------
# API ENDPOINTS 2, 3, 4 (Improvement Logic)
# ---------------------------------------------------------
# Note: Apply the same try/except and parse_ai_response logic to your other endpoints
# to ensure the server never crashes during training or file reading.

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)