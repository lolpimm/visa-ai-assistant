import requests
import json

# The URL of your local server's endpoint
url = "http://127.0.0.1:5000/improve-ai-manually"

# The data we want to send
data = {
    "instructions": "Be incredibly enthusiastic and use lots of emojis."
}

print("Sending request to server... This might take a few seconds as Gemini rewrites the prompt.")

# Send the POST request
response = requests.post(url, json=data)

# Print the result
print("\n--- SERVER RESPONSE ---")
print(json.dumps(response.json(), indent=2))