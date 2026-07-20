import requests

url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Golden_Gate_Bridge_%28cropped%29.jpg/800px-Golden_Gate_Bridge_%28cropped%29.jpg"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
r = requests.get(url, headers=headers)
print(r.status_code, r.headers.get('Content-Type'))
print(r.content[:100])
