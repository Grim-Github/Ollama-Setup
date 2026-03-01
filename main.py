import ollama_api_06

sites = ollama_api_06.chat_with_ollama_json(
    "Hello! Find 5 interesting sites. Return a JSON array containing only the links."
)

print(f"Found {len(sites)} sites:")
for site in sites:
    print(site)
