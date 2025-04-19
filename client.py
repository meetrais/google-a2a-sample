# client_simple.py
import requests
import uuid
import json

SERVER_URL = "http://127.0.0.1:5000" # URL of your A2A server

try:
    agent_card_url = f"{SERVER_URL}/.well-known/agent.json"
    response = requests.get(agent_card_url)
    response.raise_for_status() # Check for HTTP errors
    agent_info = response.json()

    task_id = str(uuid.uuid4())
    task_payload = {
        "id": task_id,
        "message": {
            "role": "user",
            "parts": [{"text": "What is Gen AI?"}] # Example question
        }
    }

    task_send_url = f"{SERVER_URL}/tasks/send"
    response = requests.post(task_send_url, json=task_payload)
    response.raise_for_status() # Check for HTTP errors

    response_data = response.json()
    agent_reply = response_data['messages'][-1]['parts'][0]['text']
    print("\n--- Agent Response ---")
    print(agent_reply)
    print("----------------------")

except requests.exceptions.RequestException as e:
    print(f"\nError communicating with server: {e}")
    print("Please ensure the server is running and accessible.")
except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
    print(f"\nError processing server response: {e}")
    print("The server response might not be in the expected A2A format.")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")