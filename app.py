#used Gemini API
from google import genai
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from google.genai import types
import os

def create_embedding(input_text):
    client = genai.Client()
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=input_text
    )
    return response.embeddings[0].values

def chat_bot():
    client = genai.Client()
    print("welcome to ChatBot! Type 'exit' to quit.")
    while True:
        input_text= input("You: ")
        if input_text.strip().lower() =="exit":
            print("ChatBot: GoodBye!")
            break
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents = input_text
        )
        print(f"ChatBot: {response.text}")
        print("\nIf you want to exit, Type 'exit'.")

def indexing(pdf_path):
        url= os.getenv("QDRANT_URL")
        api_key=os.getenv("QDRANT_API_KEY")
    
        loader = PyPDFLoader(file_path=pdf_path)
        docs =loader.load()
        print("document loaded.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size =800,
            chunk_overlap = 100
        )
        chunks = text_splitter.split_documents(documents=docs)
        print("Text splited into chucks.")

        embedding_model = GoogleGenerativeAIEmbeddings(
            model = 'models/gemini-embedding-001'
        )
        
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embedding_model,
            timeout=60.0,
            prefer_grpc =True,
            url= url,
            api_key=api_key,
            collection_name="learn_rag1"
        )
        print("Vector stored.")
    
def retrieval():

    embedding_model = GoogleGenerativeAIEmbeddings(
            model = 'models/gemini-embedding-001'
        )
    # qdrant_client= QdrantClient(
    #     url= os.getenv("QDRANT_URL"),
    #     api_key=os.getenv("QDRANT_API_KEY")
    # )
    
    vector_db = QdrantVectorStore.from_existing_collection(
        url =os.getenv("QDRANT_URL"),
        collection_name="learn_rag1",
        embedding=embedding_model,
        api_key=os.getenv("QDRANT_API_KEY")
    )

    user_query = input("Ask qustions: ")
    search_result = vector_db.similarity_search(query=user_query)

    context = [f"\n\n\n".join([f"Page content: {result.page_content} \nPage number:{result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_result])]
    
    SYSTEM_PROMPT =f"""
    You are a helpfull AI assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page numbers.
    You should only answer the user based on the following context and navigate the user to open the right page number to know more.
    Context:{context}
    """

    client = genai.Client()
    
    result =client.models.generate_content(
        model="gemini-2.5-flash",
        config= types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        ),
        contents =user_query
    )
    print(result.text)

if __name__ == "__main__":

    load_dotenv()

    pdf_path =Path(__file__).parent / "data/Cricket_with_AI_Product_Development_Document.pdf"
    #chat_bot()
    #print(create_embedding(input("enter Embedding text: "))[:5])

    indexing(pdf_path=pdf_path)
    retrieval()

