import ollama

def chat_with_ollama(user_input, system_prompt="You are a helpful and concise assistant.", model="gpt-oss:20b-cloud"):
    """
    Sends a message to the Ollama API and returns the AI's response.
    
    Args:
        user_input (str): The message from the user.
        system_prompt (str): The system prompt to set the AI's behavior.
        model (str): The name of the AI model to use.
        
    Returns:
        str: The AI's response content.
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
        
    messages.append({"role": "user", "content": user_input})
    
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]

if __name__ == "__main__":
    # Example usage when running the script directly
    user_msg = input("You: ")
    reply = chat_with_ollama(user_msg)
    print(f"AI: {reply}")
