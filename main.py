from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langchain_community.vectorstores import FAISS


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

def indexing(file_path):

    #document loader
    load_pdf =PyPDFLoader(file_path=file_path)
    docs =load_pdf.load()
    print("document loaded...")

    #chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400
    )
    chunks =text_splitter.split_documents(documents=docs)
    print("document splitted into chunks...")

    #embedding
    # embedding = OpenAIEmbeddings(
    #     model = "gemini-embedding-001"
    # )
    embedding = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )
    

    vector_store =FAISS.from_documents(
        documents=chunks,
        embedding=embedding
    )

    #Store a documents chunk into vector store
    # vector_store = QdrantVectorStore.from_documents(
    #     documents=chunks,
    #     embedding=embedding,
    #     url =os.getenv("QDRANT_URL"),
    #     api_key =os.getenv("QDRANT_API_KEY"),
    #     collection_name ="learn_genAI",
    #     timeout=60.0,
    #     prefer_grpc =True
    # )

    print("indexing of document is done....")
    
    return vector_store

#not used
def add_inteligence(contexts, quary):
    SYSTEM_PROMPT =f"""
    You are a helpfull ai assistent who anwser user quary based on the available context retraived from the PDF along with page content and page number
    You should only only anwser user based on the following context and navigate user to open right page to know more details.
    \n\nContexts: '{contexts}'.
    """
    result = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
        {
            "role":"system",
            "content": SYSTEM_PROMPT
        },
        {
            "role":"user",
            "content": quary
        }]

    )
    return result.choices[0].message.content

def retrieval(vector_db):
    retriever=vector_db.as_retriever()
    print("Document Q&A ChatBot Ready!")

    while True:
        input_query = input("\nAsk a question (or type 'exit'): ")

        if input_query.strip().lower() =="exit":
            print("ChatBot: bye.")
            break

        results =retriever.invoke(input_query)

        ## added inteligence using Gemini AI
        contexts =[f"\n\n Page content: {result.page_content} \n\nPage nuumber:{result.metadata["page_label"]} \n\nFile location: {result.metadata["source"]}" for result in results]
        SYSTEM_PROMPT =f"""
        You are a helpfull ai assistent who anwser user quary based on the available context retraived from the PDF along with page content and page number
        You should only only anwser user based on the following context and navigate user to open right page to know more details.
        \n\nContexts: '{contexts}'.
        """
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
            {
                "role":"system",
                "content": SYSTEM_PROMPT
            },
            {
                "role":"user",
                "content": input_query
            }]

        )
        print("ChatBot: ",response.choices[0].message.content)


def doc_rag():
    client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    #rag 
    file_path = Path(__file__).parent/"data/Cricket_with_AI_Product_Development_Document.pdf"
    vector_db = indexing(file_path=file_path)
    retrieval(vector_db=vector_db)

if __name__ == "__main__":

    load_dotenv()
    
    #chat_bot()
    #create_embedding(input("Enter you text: "))
    doc_rag()


    