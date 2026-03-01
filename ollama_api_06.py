"""
This script demonstrates how to interact with Ollama's cloud models.
Note: To use cloud models, you must first sign in via the Ollama CLI:
    ollama signin
"""

try:
    import ollama
except ImportError:
    print("Error: The 'ollama' library is not installed.")
    print("Please install it using: pip install ollama")
    exit(1)


def chat_with_cloud_model(prompt, model="gpt-oss:120b-cloud"):
    """
    Sends a message to the gpt-oss:20b-cloud model using the Ollama Python library.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error communicating with Ollama: {e}"


if __name__ == "__main__":
    print(f"--- Ollama Cloud Interface ({'gpt-oss:120b-cloud'}) ---")
    print("Make sure you have run 'ollama signin' in your terminal.\n")

    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        exit()

    print("\nModel thinking...")
    reply = chat_with_cloud_model(user_input)

    print("\nGPT-OSS:20B-CLOUD:")
    print(reply)
