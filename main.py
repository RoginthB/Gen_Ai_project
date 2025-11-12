from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def ChatBot():
    
    client = OpenAI(
    
    base_url="https://generativelanguage.googleapis.com/v1beta/")

    print("welcome to ChatBot! Type 'exit' to quit.")

    while True:
        text = input("You: ")

        if text.strip().lower()=="exit":
            print("ChatBot: Goodbye!")
            break
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{
                "role": "system",
                "content": "You are a helful assistant."
             },{"role": "user", "content": text}
             ]
        )
        print("ChatBot: ", response.choices[0].message.content)
        print("\nIf you want to exit, type 'exit'.")



if __name__ == "__main__":
    ChatBot()