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
# Reverting back to your original model name
MODEL_NAME = 'gemini-2.5-flash' 

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
        cur.execute("INSERT INTO ai_prompts (prompt_text) VALUES (%s) RETURNING id;", (new_prompt_text,))
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
# API ENDPOINT 1: GENERATE REPLY (Uses DB Prompt)
# ---------------------------------------------------------
@app.route('/generate-reply', methods=['POST'])
def generate_reply():
    try:
        data = request.json
        client_sequence = data.get('clientSequence', '')
        chat_history = data.get('chatHistory', [])

        current_prompt = get_current_prompt()
        
        # Limit history to avoid token bloat/quota issues
        formatted_history = "CHAT HISTORY:\n"
        for msg in chat_history[-5:]:
            formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
        
        full_prompt = f"{current_prompt}\n\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{client_sequence}\n\nReturn your answer in JSON format: {{\"reply\": \"...\"}}"

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=full_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        reply = parse_ai_response(response.text, 'reply')
        return jsonify({"aiReply": reply})

    except Exception as e:
        error_msg = str(e)
        print(f"Generate Reply Error: {error_msg}")
        if "429" in error_msg:
            return jsonify({"aiReply": "Server is busy (Quota hit). Please wait a moment."}), 429
        return jsonify({"aiReply": f"Error: {error_msg}"}), 500

# ---------------------------------------------------------
# API ENDPOINT 2: AUTO-IMPROVE AI (Single Loop)
# ---------------------------------------------------------
@app.route('/improve-ai', methods=['POST'])
def improve_ai():
    try:
        data = request.json
        client_sequence = data.get('clientSequence', '')
        chat_history = data.get('chatHistory', [])
        actual_reply = data.get('consultantReply', '')
        
        current_prompt = get_current_prompt()
        
        formatted_history = "CHAT HISTORY:\n"
        for msg in chat_history[-5:]:
            formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
        
        prediction_prompt = f"{current_prompt}\n\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{client_sequence}\n\nReturn your answer in JSON format: {{\"reply\": \"...\"}}"
        
        pred_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prediction_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        predicted_reply = parse_ai_response(pred_response.text, 'reply')

        editor_meta_prompt = f"""
        You are an expert Prompt Engineer. Your job is to improve an AI chatbot prompt.
        
        CURRENT PROMPT:
        {current_prompt}
        
        CONTEXT (Client Message): {client_sequence}
        
        THE ACTUAL HUMAN CONSULTANT REPLIED: {actual_reply}
        THE AI PREDICTED THIS INSTEAD: {predicted_reply}
        
        Analyze why the AI's prediction was different from the human consultant's actual reply. 
        Rewrite the CURRENT PROMPT to fix these issues. Make surgical updates so the AI acts more like the human consultant.
        
        Output ONLY valid JSON in this exact format:
        {{"prompt": "the completely rewritten new prompt text goes here"}}
        """

        editor_response = client.models.generate_content(
            model=MODEL_NAME,
            contents=editor_meta_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        updated_prompt = parse_ai_response(editor_response.text, 'prompt')
        save_new_prompt(updated_prompt)

        return jsonify({
            "predictedReply": predicted_reply,
            "updatedPrompt": updated_prompt
        })
    except Exception as e:
        print(f"Improve AI Error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# API ENDPOINT 3: MANUALLY IMPROVE AI
# ---------------------------------------------------------
@app.route('/improve-ai-manually', methods=['POST'])
def improve_ai_manually():
    try:
        data = request.json
        instructions = data.get('instructions', '')
        
        current_prompt = get_current_prompt()
        
        manual_editor_prompt = f"""
        You are a Prompt Engineer updating an existing prompt based on human instructions.
        
        CURRENT PROMPT:
        {current_prompt}
        
        HUMAN INSTRUCTIONS:
        {instructions}
        
        Rewrite the CURRENT PROMPT to perfectly incorporate these instructions.
        
        Output ONLY valid JSON in this exact format:
        {{"prompt": "the newly updated prompt text goes here"}}
        """

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=manual_editor_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        
        updated_prompt = parse_ai_response(response.text, 'prompt')
        save_new_prompt(updated_prompt)

        return jsonify({"updatedPrompt": updated_prompt})
    except Exception as e:
        print(f"Manual Improve Error: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------
# API ENDPOINT 4: BATCH TRAIN ON FILE 
# ---------------------------------------------------------
@app.route('/train-on-file', methods=['POST'])
def train_on_file():
    try:
        with open('conversations.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        training_logs = []
        
        # Train on the first 3 conversations to keep the demo fast
        for convo in data[:3]:
            messages = convo.get('conversation', [])
            scenario = convo.get('scenario', 'Unknown Scenario')
            
            if len(messages) >= 4:
                chat_history = []
                for msg in messages[:3]:
                    role = "client" if msg["direction"] == "in" else "consultant"
                    chat_history.append({"role": role, "message": msg["text"]})
                
                client_sequence = messages[2]["text"]
                actual_reply = messages[3]["text"]
                
                current_prompt = get_current_prompt()
                
                formatted_history = "CHAT HISTORY:\n"
                for msg in chat_history:
                    formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
                
                prediction_prompt = f"{current_prompt}\n\nSCENARIO: {scenario}\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{client_sequence}\n\nReturn your answer in JSON format: {{\"reply\": \"...\"}}"
                
                pred_response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prediction_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                predicted_reply = parse_ai_response(pred_response.text, 'reply')
                
                editor_meta_prompt = f"""
                You are an expert Prompt Engineer. Your job is to improve an AI chatbot prompt based on real human data.
                
                CURRENT PROMPT:
                {current_prompt}
                
                SCENARIO: {scenario}
                CONTEXT (Client Message): {client_sequence}
                
                THE ACTUAL HUMAN CONSULTANT REPLIED: {actual_reply}
                THE AI PREDICTED THIS INSTEAD: {predicted_reply}
                
                Analyze the differences. Rewrite the CURRENT PROMPT to incorporate the human's style and logic.
                
                Output ONLY valid JSON in this exact format:
                {{"prompt": "the completely rewritten new prompt text goes here"}}
                """

                editor_response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=editor_meta_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                updated_prompt = parse_ai_response(editor_response.text, 'prompt')
                save_new_prompt(updated_prompt)
                
                training_logs.append({
                    "scenario": scenario,
                    "predicted": predicted_reply,
                    "actual": actual_reply,
                    "resulting_prompt_update": updated_prompt
                })
                
        return jsonify({"message": "Training Complete!", "logs": training_logs})
        
    except Exception as e:
        print(f"Train on File Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)