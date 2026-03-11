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

# Initialize Flask and the NEW Gemini SDK
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')
client = genai.Client(api_key=api_key)

# ---------------------------------------------------------
# DATABASE HELPER FUNCTIONS
# ---------------------------------------------------------
def get_db_connection():
    """Connects to the Neon Serverless Postgres Database."""
    return psycopg2.connect(db_url)

def get_current_prompt():
    """Fetches the latest AI prompt from the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT prompt_text FROM ai_prompts ORDER BY id DESC LIMIT 1;")
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else "Default Prompt Fallback"

def save_new_prompt(new_prompt_text):
    """Saves a newly improved prompt into the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO ai_prompts (prompt_text) VALUES (%s) RETURNING id;",
        (new_prompt_text,)
    )
    conn.commit()
    cur.close()
    conn.close()

# ---------------------------------------------------------
# API ENDPOINT 1: GENERATE REPLY (Uses DB Prompt)
# ---------------------------------------------------------
@app.route('/generate-reply', methods=['POST'])
def generate_reply():
    data = request.json
    client_sequence = data.get('clientSequence', '')
    chat_history = data.get('chatHistory', [])

    # 1. Get the LIVE prompt from the database
    current_prompt = get_current_prompt()

    # 2. Format the history
    formatted_history = "CHAT HISTORY:\n"
    for msg in chat_history:
        formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
    
    full_prompt = f"{current_prompt}\n\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{client_sequence}\n\nAI REPLY:"

    # 3. Call Gemini
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=full_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    # 4. Safely parse and return
    clean_text = response.text.replace('```json', '').replace('```', '').strip()
    return jsonify({"aiReply": json.loads(clean_text).get('reply', 'Error')})

# ---------------------------------------------------------
# API ENDPOINT 2: AUTO-IMPROVE AI (Single Loop)
# ---------------------------------------------------------
@app.route('/improve-ai', methods=['POST'])
def improve_ai():
    data = request.json
    client_sequence = data.get('clientSequence', '')
    chat_history = data.get('chatHistory', [])
    actual_reply = data.get('consultantReply', '')
    
    current_prompt = get_current_prompt()
    
    formatted_history = "CHAT HISTORY:\n"
    for msg in chat_history:
        formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
    
    prediction_prompt = f"{current_prompt}\n\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{client_sequence}\n\nAI REPLY:"
    
    pred_response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prediction_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    clean_pred_text = pred_response.text.replace('```json', '').replace('```', '').strip()
    predicted_reply = json.loads(clean_pred_text).get('reply', '')

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
        model='gemini-2.5-flash',
        contents=editor_meta_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    clean_editor_text = editor_response.text.replace('```json', '').replace('```', '').strip()
    updated_prompt = json.loads(clean_editor_text).get('prompt', current_prompt)
    
    save_new_prompt(updated_prompt)

    return jsonify({
        "predictedReply": predicted_reply,
        "updatedPrompt": updated_prompt
    })

# ---------------------------------------------------------
# API ENDPOINT 3: MANUALLY IMPROVE AI
# ---------------------------------------------------------
@app.route('/improve-ai-manually', methods=['POST'])
def improve_ai_manually():
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
        model='gemini-2.5-flash',
        contents=manual_editor_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    clean_text = response.text.replace('```json', '').replace('```', '').strip()
    updated_prompt = json.loads(clean_text).get('prompt', current_prompt)
    
    save_new_prompt(updated_prompt)

    return jsonify({"updatedPrompt": updated_prompt})

# ---------------------------------------------------------
# API ENDPOINT 4: BATCH TRAIN ON FILE (The A+ Feature)
# ---------------------------------------------------------
@app.route('/train-on-file', methods=['POST'])
def train_on_file():
    """Reads the real conversations.json and trains the AI prompt automatically."""
    try:
        # utf-8 encoding prevents Windows from crashing when reading emojis!
        with open('conversations.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        training_logs = []
        
        # Train on the first 3 conversations to keep the demo fast
        for convo in data[:3]:
            messages = convo.get('conversation', [])
            scenario = convo.get('scenario', 'Unknown Scenario')
            
            if len(messages) >= 4:
                chat_history = []
                # Grab the first 3 messages for context
                for msg in messages[:3]:
                    role = "client" if msg["direction"] == "in" else "consultant"
                    chat_history.append({"role": role, "message": msg["text"]})
                
                client_sequence = messages[2]["text"]
                actual_reply = messages[3]["text"]
                
                current_prompt = get_current_prompt()
                
                formatted_history = "CHAT HISTORY:\n"
                for msg in chat_history:
                    formatted_history += f"({msg['role'].upper()}) {msg['message']}\n"
                
                prediction_prompt = f"{current_prompt}\n\nSCENARIO: {scenario}\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{client_sequence}\n\nAI REPLY:"
                
                pred_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prediction_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                # Safely parse JSON
                clean_pred_text = pred_response.text.replace('```json', '').replace('```', '').strip()
                predicted_reply = json.loads(clean_pred_text).get('reply', '')
                
                editor_meta_prompt = f"""
                You are an expert Prompt Engineer. Your job is to improve an AI chatbot prompt based on real human data.
                
                CURRENT PROMPT:
                {current_prompt}
                
                SCENARIO: {scenario}
                CONTEXT (Client Message): {client_sequence}
                
                THE ACTUAL HUMAN CONSULTANT REPLIED: {actual_reply}
                THE AI PREDICTED THIS INSTEAD: {predicted_reply}
                
                Analyze the differences. The human consultant's reply is the "gold standard". 
                Rewrite the CURRENT PROMPT to incorporate the human's style, logic, and factual knowledge.
                
                Output ONLY valid JSON in this exact format:
                {{"prompt": "the completely rewritten new prompt text goes here"}}
                """

                editor_response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=editor_meta_prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                
                clean_editor_text = editor_response.text.replace('```json', '').replace('```', '').strip()
                updated_prompt = json.loads(clean_editor_text).get('prompt', current_prompt)
                
                save_new_prompt(updated_prompt)
                
                training_logs.append({
                    "scenario": scenario,
                    "predicted": predicted_reply,
                    "actual": actual_reply,
                    "resulting_prompt_update": updated_prompt
                })
                
        return jsonify({"message": "Training Complete!", "logs": training_logs})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)