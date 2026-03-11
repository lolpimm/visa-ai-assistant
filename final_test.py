import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def test_generate():
    print("\n--- Testing Response Generation ---")
    payload = {
        "clientSequence": "I am a software engineer from the UK. Can I apply for the DTV?",
        "chatHistory": []
    }
    res = requests.post(f"{BASE_URL}/generate-reply", json=payload)
    print(f"AI Response: {res.json().get('aiReply')}")

def test_manual_learning():
    print("\n--- Testing Manual Learning (Prompt Update) ---")
    payload = {"instructions": "Always mention that we have a 100% success rate for UK engineers."}
    res = requests.post(f"{BASE_URL}/improve-ai-manually", json=payload)
    print("Prompt updated successfully.")

if __name__ == "__main__":
    # 1. See how it responds now
    test_generate()
    
    # 2. Tell it to 'learn' a new rule
    test_manual_learning()
    
    # 3. See if it actually learned it
    print("\n--- Testing if AI learned the new rule ---")
    test_generate()