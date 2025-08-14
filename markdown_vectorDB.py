import os
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

EMBEDDINGS_OPENAI_API_TYPE = ""
EMBEDDINGS_OPENAI_MODEL = ""
EMBEDDINGS_OPENAI_API_BASE = ""
EMBEDDINGS_OPENAI_API_KEY = ""
EMBEDDINGS_AZURE_DEPLOYMENT_NAME = ""

class MarkdownIngestor:
    def __init__(self, markdown_file_path):
        self.markdown_file_path = markdown_file_path
        self.vector_db_path = "band3_and_below_db_markdown_11_Aug"

    def load_markdown(self):
        with open(self.markdown_file_path, "r", encoding="utf-8") as f:
            return f.read()

    def split_markdown(self, markdown_text):
        headers_to_split_on = [
            ("##", "Question"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        split_docs = splitter.split_text(markdown_text)

        combined_docs = []
        for doc in split_docs:
            question = doc.metadata.get("Question", "").strip()
            answer = doc.page_content.strip()
            content = f"{question}\n{answer}"
            combined_docs.append(Document(page_content=content, metadata=doc.metadata))

        return combined_docs

    def create_vector_db(self, docs):
        embeddings = AzureOpenAIEmbeddings(
            deployment=EMBEDDINGS_AZURE_DEPLOYMENT_NAME,
            model=EMBEDDINGS_OPENAI_MODEL,
            azure_endpoint=EMBEDDINGS_OPENAI_API_BASE,
            openai_api_type=EMBEDDINGS_OPENAI_API_TYPE,
            api_key=EMBEDDINGS_OPENAI_API_KEY,
            chunk_size=64
        )
        vectordb = FAISS.from_documents(documents=docs, embedding=embeddings)
        vectordb.save_local(self.vector_db_path)

    def ingest(self):
        print("🔹 Loading Markdown...")
        markdown_text = self.load_markdown()

        print("🔹 Splitting using MarkdownHeaderTextSplitter...")
        docs = self.split_markdown(markdown_text)
        print(f"🔹 Created {len(docs)} chunks")

    # ✅ Save the chunks to a file for validation
        with open("split_chunks.txt", "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                f.write(f"--- Chunk {i + 1} ---\n")
                f.write(doc.page_content)
                f.write("\n\n")

        print("🔹 Creating Vector DB...")
        self.create_vector_db(docs)
        print("✅ Vector DB created and saved locally!")
        print("📁 Chunks saved to 'split_chunks.txt' for validation.")

if __name__ == "__main__":
    md_path = os.path.join(os.path.dirname(__file__), "Band_3_and_Below New 1 1 (1).md")
    ingestor = MarkdownIngestor(md_path)
    ingestor.ingest()
