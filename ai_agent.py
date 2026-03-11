import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("LLM_API_KEY")

# Configure the Gemini client
genai.configure(api_key=api_key)
# We use gemini-2.5-flash because it is the fastest and most cost-effective for chatbots
model = genai.GenerativeModel('gemini-2.5-flash')

# ---------------------------------------------------------
# THE BASE PROMPT (What makes the AI act like a human)
# ---------------------------------------------------------
BASE_PROMPT = """
You are a friendly, human immigration consultant helping a client with Thai visas.
Your goal is to be helpful, casual, and NOT sound like a robot. 
Read the chat history to understand the context, then reply to the client's latest message.

Output your response ONLY in valid JSON format like this:
{"reply": "your casual, helpful response goes here"}
"""

def generate_ai_reply(chat_history_list, latest_client_message):
    """
    Takes the chat history and the newest message, sends it to Gemini, 
    and returns the AI's predicted reply.
    """
    # 1. Format the history so the AI can read it easily
    formatted_history = "CHAT HISTORY:\n"
    for msg in chat_history_list:
        formatted_history += f"({msg['role'].upper()}) {msg['content']}\n"
    
    # 2. Combine the prompt, history, and the new message
    full_prompt = f"{BASE_PROMPT}\n\n{formatted_history}\nLATEST CLIENT MESSAGE:\n{latest_client_message}\n\nAI REPLY:"
    
    # 3. Ask Gemini for the response and force it to return JSON
    response = model.generate_content(
        full_prompt,
        generation_config=genai.GenerationConfig(response_mime_type="application/json")
    )
    
    # 4. Convert the text response into a Python dictionary
    return json.loads(response.text)

# ---------------------------------------------------------
# TEST SCRIPT (Runs only if you execute this file directly)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Load our sample data
    with open('conversations.json', 'r') as file:
        data = json.load(file)
    
    # Grab the first conversation
    conversation = data[0]["messages"]
    
    # Let's pretend the chat history is the first two messages
    history = conversation[0:2]
    # And the client just sent the 3rd message
    new_message = conversation[2]["content"] 
    
    print("Testing AI Agent...")
    print(f"Client said: {new_message}")
    
    # Run our function!
    result = generate_ai_reply(history, new_message)
    print(f"\nAI Consultant Replied: {result['reply']}")