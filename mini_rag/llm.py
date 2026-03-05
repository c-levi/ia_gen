import requests

def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]
