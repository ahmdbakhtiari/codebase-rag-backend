from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DOCUMENT_PATH = BASE_DIR / "formulas.py"
CHROMA_PATH = BASE_DIR / "chromadb"

with open(DOCUMENT_PATH, "r", encoding="utf-8") as document:
    py_file = document.read()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)

documents = text_splitter.create_documents([py_file])

texts = [doc.page_content for doc in documents]

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(texts)


client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="codebase"
)


collection.add(
    ids=[str(i) for i in range(len(texts))],
    documents=texts,
    embeddings=embeddings.tolist()
)

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)



def generate_answer(question: str):

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    retrieved_chunks = results["documents"][0]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Context:
{context}

Question:
{question}

Answer:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    outputs = llm.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

    answer = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return answer