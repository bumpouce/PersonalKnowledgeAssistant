import requests

url = 'http://localhost:11434/api/generate'
data = {
    "model": "gemma3:4b-it-qat",
    "prompt": "Why is the sky blue?",
    "stream": False
}
r = requests.post(url, json=data)
print(r.json()["response"])

# curl.exe http://localhost:11434/api/generate -d '{\"model\": \"gemma3:4b-it-qat\", \"prompt\": \"Why is the sky blue?\", \"stream\": false}'
# >>> import requests
# >>> r = requests.get('https://httpbin.org/basic-auth/user/pass', auth=('user', 'pass'))
# >>> r.status_code
# 200
# >>> r.headers['content-type']
# 'application/json; charset=utf8'
# >>> r.encoding
# 'utf-8'
# >>> r.text
# '{"authenticated": true, ...'
# >>> r.json()
# {'authenticated': True, ...}