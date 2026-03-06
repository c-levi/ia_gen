import requests

def ask_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "mistral", #"llama3.2:3b", 
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=data)
    return response.json()["response"]
