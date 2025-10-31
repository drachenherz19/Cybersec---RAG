import argparse
from time import sleep
from typing import List, Dict, Any

import openai
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import Session

from config import OPENAI_API_KEY
from db.connection import SessionLocal
from db.models import CVE


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class CyberSecurityRAG:
    def __init__(self, use_openai_embeddings: bool = False):
        self.use_openai_embeddings = use_openai_embeddings
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None

    def load_db_data(self, db: Session) -> List[Document]:
        """
        Load and process CVE data from the database.
        """
        cves = db.query(CVE).all()
        documents = []
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,  # Token limit per chunk
            chunk_overlap=200,
            separator="\n",
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=True,
        )

        for cve in cves:
            content = f"""
            CVE ID: {cve.cve_id}
            Published Date: {cve.published_date}
            Last Modified: {cve.last_modified_date}
            Description: {cve.description}
            CVSS Score: {cve.cvss_score}

            Additional Metadata:
            {cve.additional_metadata}
            """
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                documents.append(Document(page_content=chunk))

        return documents

    def initialize_embeddings(self, openai_api_key: str = None):
        """Initialize the embedding model."""
        if self.use_openai_embeddings:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def create_vector_store(self, documents: List[Document], save_path: str = None):
        """Create and optionally save FAISS vector store."""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call `initialize_embeddings` first.")

        token_limit = 8192
        chunks = []
        for doc in documents:
            if len(doc.page_content) > token_limit:
                text_splitter = CharacterTextSplitter(
                    chunk_size=token_limit,
                    chunk_overlap=200,
                    separator="\n",
                    length_function=len,
                    is_separator_regex=False,
                    strip_whitespace=True,
                )
                chunks.extend(text_splitter.split_text(doc.page_content))
            else:
                chunks.append(doc.page_content)

        print(f"Total chunks created: {len(chunks)}")

        # Batch process embeddings to avoid rate limits
        batch_size = 10
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            try:
                embeddings.extend(self._embed_documents_with_retry(batch))
            except Exception as e:
                print(f"Error embedding batch {i}: {e}")

        self.vector_store = FAISS.from_texts(chunks, self.embeddings)

        if save_path:
            self.vector_store.save_local(save_path)
            print(f"Vector store saved to {save_path}")

    def _embed_documents_with_retry(self, texts: List[str], max_retries: int = 5, backoff: int = 2):
        """Embed documents with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.embeddings.embed_documents(texts)
            except openai.error.RateLimitError:
                print(f"Rate limit hit. Retrying in {backoff} seconds... (Attempt {attempt + 1}/{max_retries})")
                sleep(backoff)
                backoff *= 2
            except Exception as e:
                print(f"Unexpected error: {e}")
                raise
        raise Exception("Maximum retries exceeded for embedding documents.")

    def load_vector_store(self, load_path: str):
        """Load existing FAISS vector store."""
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call `initialize_embeddings` first.")

        try:
            import faiss
            faiss.allow_dangerous_deserialization = True
            self.vector_store = FAISS.load_local(load_path, self.embeddings)
            print("Vector store loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading vector store: {e}")
            print("Creating a new vector store instead...")
            return False

    def initialize_qa_chain(self, openai_api_key: str):
        """Initialize the QA chain with ChatGPT."""
        if not self.vector_store:
            raise ValueError("Vector store not initialized.")

        self.llm = ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo-16k",
            openai_api_key=openai_api_key,
        )

        template = """You are a cybersecurity expert assistant. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}

        Answer: Let me help you with that based on the CVE database information."""

        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        prompt = PromptTemplate.from_template(template)
        self.qa_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[Any, Any]:
        """Query the system with a security-related question."""
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call `initialize_qa_chain` first.")

        docs = self.vector_store.similarity_search(question, k=5)
        answer = self.qa_chain.invoke(question)

        return {"answer": answer, "source_documents": docs}


def main():
    parser = argparse.ArgumentParser(description="Cybersecurity RAG System")
    parser.add_argument("--vector_store_path", type=str, help="Path to save/load FAISS vector store")
    parser.add_argument("--use_openai_embeddings", action="store_true", help="Use OpenAI embeddings instead of HuggingFace")
    args = parser.parse_args()

    try:
        rag = CyberSecurityRAG(use_openai_embeddings=args.use_openai_embeddings)
        rag.initialize_embeddings(OPENAI_API_KEY)

        print("Fetching CVE data from the database...")
        db = SessionLocal()
        try:
            documents = rag.load_db_data(db)
            if not documents:
                print("No CVE data found in the database.")
                return

            if args.vector_store_path and rag.load_vector_store(args.vector_store_path):
                print("Vector store loaded successfully.")
            else:
                print("Creating a new vector store...")
                rag.create_vector_store(documents, save_path=args.vector_store_path)

        finally:
            db.close()

        rag.initialize_qa_chain(OPENAI_API_KEY)
        print("\nCybersecurity RAG System initialized. Enter your questions (type 'exit' to quit):")

        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() == "exit":
                print("Exiting system. Goodbye!")
                break

            result = rag.query(question)
            print("\nAnswer:", result["answer"])
            if input("\nSee source documents? (y/n): ").lower() == "y":
                for i, doc in enumerate(result["source_documents"], 1):
                    print(f"Source {i}:\n{doc.page_content}")

    except Exception as e:
        print(f"Error initializing RAG system: {e}")


if __name__ == "__main__":
    main()
