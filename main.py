from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter



def create_embedding(input_text):
    response =client.embeddings.create(
        input=input_text,
        model="gemini-embedding-001"
    )
    print(response.data[0].embedding[:5])

    return response

def chat_bot():
    
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

    load_dotenv()
    pdf_path =Path(__file__).parent / "data/NXOpen_Getting_Started.pdf"
    client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/")
    
    #chat_bot()
    #create_embedding(input("Enter you text: "))

    loader = PyPDFLoader(file_path=pdf_path)
    docs =loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =1000,
        chunk_overlap = 400
    )
    chunks = text_splitter.split_documents(documents=docs)

    