import json
import os
import ollama

def load_config(file_path="config.json"):
    """Loads configuration from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return {}

# Load global defaults
CONFIG = load_config()
DEFAULT_MODEL = CONFIG.get("model", "gpt-oss:120b-cloud")
DEFAULT_SYSTEM_PROMPT = CONFIG.get("system_prompt", "You are a helpful and concise assistant.")
DEFAULT_JSON_SYSTEM_PROMPT = CONFIG.get("json_system_prompt", "You are a helpful assistant that always responds in valid JSON.")


def chat_with_ollama(
    user_input,
    system_prompt=None,
    model=None,
):
    """
    Sends a message to the Ollama API and returns the AI's response.
    """
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    model = model or DEFAULT_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_input})

    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def chat_with_ollama_json(
    user_input,
    system_prompt=None,
    model=None,
):
    """
    Sends a message to the Ollama API and returns a parsed JSON response.
    """
    system_prompt = system_prompt or DEFAULT_JSON_SYSTEM_PROMPT
    model = model or DEFAULT_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_input})

    # The 'format' parameter set to 'json' forces the model to output valid JSON
    response = ollama.chat(model=model, messages=messages, format="json")
    content = response["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        return {"error": "Invalid JSON returned", "raw_content": content}


if __name__ == "__main__":
    # Example usage for normal chat
    print("--- Normal Chat ---")
    user_msg = "Tell me a short joke."
    reply = chat_with_ollama(user_msg)
    print(f"AI: {reply}\n")

    # Example usage for JSON response
    print("--- JSON Response ---")
    json_prompt = "Give me a list of 3 fruits and their colors in JSON format."
    json_reply = chat_with_ollama_json(json_prompt)
    print(f"AI (JSON): {json.dumps(json_reply, indent=4)}")
