import requests

url = 'http://localhost:11434/api/generate'
data = {
    "model": "gemma3:4b-it-qat",
    "prompt": "Why is the sky blue?",
    "stream": False
}
r = requests.post(url, json=data)
print(r.json()["response"])