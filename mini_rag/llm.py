import requests

def ask_ollama(prompt):

    model = "phi3:mini" # "mistral", #"llama3.2:3b", 

    print(f"Model used: {model}")
    print(f"Prompt length (chars): {len(prompt)}")

    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            # "num_predict": 100,   # limite la taille de réponse
            "temperature": 0.2
        }
    }
    response = requests.post(url, json=data)
    return response.json()["response"]
