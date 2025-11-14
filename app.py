from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embeddings
embeddings = download_hugging_face_embeddings()

# Load Pinecone index
index_name = "ai-medical"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5 })

# Load local model
print("Loading model...")
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,
    #max_length=1024,
    max_length=512,
    do_sample=False,
)
print("Model loaded!")

def generate_answer(question):
    # Retrieve relevant docs
    docs = retriever.invoke(question)
    
    # Build context from retrieved docs
    context = "\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt_text = f"""You are an expert Medical assistant for question-answering tasks.
    Use ONLY the retrieved context to answer the question.
    If the answer is not in the context, respond with "I don't know".
    Provide a clear, detailed answer (max 200 words).

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    # Generate answer
    result = pipe(prompt_text, max_length=512, do_sample=False)
    answer = result[0]['generated_text'].split("Answer:")[-1].strip()
    
    # Extract book name and page number from source
    sources_info = []
    seen = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        # Extract filename from full path
        filename = source.split("\\")[-1] if "\\" in source else source.split("/")[-1]
        page = doc.metadata.get("page", "")
        
        source_str = f"{filename} (Page {page})" if page else filename
        if source_str not in seen:
            sources_info.append(source_str)
            seen.add(source_str)
    
    return answer, sources_info

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(f"User input: {msg}")
    try:
        answer, sources_info = generate_answer(msg)
        # Format sources as standard GPT style
        sources_text = ", ".join(sources_info)
        response_text = f"{answer}\n\n[Sources: {sources_text}]"
        print(f"Response: {response_text}")
        return str(response_text)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)