import requests
import time

def test_chat():
    url = "http://localhost:8000/api/chat"
    payload = {
        "message": "Hello, how are you?",
        "history": []
    }
    
    # Wait for the backend to be fully started
    print("Waiting for backend...")
    time.sleep(5)
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error connecting to backend: {e}")

if __name__ == "__main__":
    test_chat()
