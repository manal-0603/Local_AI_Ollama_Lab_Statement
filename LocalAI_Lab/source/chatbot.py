"""Console chatbot with conversation memory (uses /api/chat)."""
import requests

OLLAMA = "http://localhost:11434"
MODEL = "qwen3.5:2b-q4_K_M"


def main():
    messages = [{"role": "system", "content":
                 "You are a helpful, concise assistant talking with a user. Answer the user directly in the "
                 "second person ('you', 'your'); never repeat the user's sentences as if they were your own. "
                 "Use the earlier messages of the conversation to answer questions about the user."}]
    print("Local AI Assistant — type 'exit' to stop.")
    while True:
        q = input("\nYou: ")
        if q.lower() == "exit":
            break
        messages.append({"role": "user", "content": q})
        r = requests.post(f"{OLLAMA}/api/chat",
                          json={"model": MODEL, "messages": messages, "stream": False,
                                "think": False,                      # no hidden reasoning: faster, no empty answers
                                "options": {"temperature": 0.3}})    # more deterministic, fewer role confusions
        r.raise_for_status()
        reply = r.json()["message"]
        messages.append({"role": "assistant", "content": reply["content"]})
        print("\nAI:", reply["content"])


if __name__ == "__main__":
    main()
