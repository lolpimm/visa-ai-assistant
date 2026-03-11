import requests
import json

print("Starting AI Training on real hackathon data...")
print("This will take about 15-30 seconds as the AI reads the JSON and rewrites its own prompt multiple times.")

# Notice how the URL is just a normal string now
response = requests.post("http://127.0.0.1:5000/train-on-file")

if response.status_code == 200:
    print("\n--- TRAINING COMPLETE ---")
    print("Here is the final, fully-trained prompt currently sitting in your database:\n")
    print(response.json()['logs'][-1]['resulting_prompt_update'])
else:
    print("\n--- ERROR ---")
    print(f"Server returned status code: {response.status_code}")
    print("Error Details:", response.text)