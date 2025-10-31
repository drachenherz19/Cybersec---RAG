import argparse
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sqlalchemy.orm import Session
from db.models import CVE
from db.connection import SessionLocal
from config import OPENAI_API_KEY


class CyberSecurityRAG:
    def __init__(self, use_openai_embeddings: bool = False):
        self.use_openai_embeddings = use_openai_embeddings
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.initial_chain = None
        self.follow_up_chain = None
        self.system_context = {}
        self.conversation_history = []

    def load_db_data(self, db: Session) -> List[Document]:
        """
        Load and process CVE data from the database.
        Args:
            db (Session): SQLAlchemy database session.
        Returns:
            List[Document]: List of processed CVE documents.
        """
        cves = db.query(CVE).all()
        documents = []
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
            length_function=len,
            is_separator_regex=False,
            strip_whitespace=True,
        )

        for cve in cves:
            content = f"""
            CVE ID: {cve.cve_id}
            Published: {cve.published_date}
            Last Modified: {cve.last_modified_date}
            Description: {cve.description}

            CVSS Score: {cve.cvss_score}

            Additional Metadata:
            {cve.additional_metadata}
            """
            chunks = text_splitter.split_text(content)
            for chunk in chunks:
                if chunk.strip():
                    documents.append(Document(page_content=chunk))

        return documents

    def initialize_embeddings(self, openai_api_key: str = None):
        if self.use_openai_embeddings:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        else:
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_vector_store(self, documents: List[Document], save_path: str = None):
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")

        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        if save_path:
            self.vector_store.save_local(save_path)
            print(f"Vector store saved to {save_path}")

    def load_vector_store(self, load_path: str):
        if not self.embeddings:
            raise ValueError("Embeddings not initialized")

        try:
            import faiss
            faiss.allow_dangerous_deserialization = True
            self.vector_store = FAISS.load_local(load_path, self.embeddings)
            print("Vector store loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False

    def initialize_qa_chain(self, openai_api_key: str):
        if not self.vector_store:
            raise ValueError("Vector store not initialized")

        self.llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-4",
            openai_api_key=openai_api_key,
        )

        initial_template = """You are a cybersecurity expert analyzing a security concern. Based on the provided context, give clear, actionable guidance.

        Security Information:
        {context}

        Current Question: {question}

        Instructions:
        Respond in clear paragraphs without any special formatting (no hashmarks, asterisks, or markdown).
        Focus on critical aspects and practical steps.
        Present information in a natural, flowing conversation style.
        Break complex concepts into digestible sections using clear transitions.
        Use simple numbering for any lists (1., 2., etc.).
        """

        follow_up_template = """You are a cybersecurity expert continuing a security analysis. Consider the complete system context.

        System Context and History:
        {chat_history}

        New Security Information:
        {context}

        Follow-up Question: {question}

        Instructions:
        Respond in clear paragraphs that:
        - Build naturally on the previous discussion
        - Include the complete system context (AWS/cloud platform, OS, etc.)
        - Connect new findings with previous issues
        - Use natural language without special formatting
        - Present information in a conversational style
        - Break down complex topics into clear sections
        Do not use any special characters, markdown, hashtags, or asterisks.
        """

        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        self.initial_chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough()
                }
                | PromptTemplate.from_template(initial_template)
                | self.llm
                | StrOutputParser()
        )

        self.follow_up_chain = (
                {
                    "context": retriever,
                    "chat_history": lambda x: self._get_formatted_history(),
                    "question": RunnablePassthrough()
                }
                | PromptTemplate.from_template(follow_up_template)
                | self.llm
                | StrOutputParser()
        )

    def query(self, question: str) -> Dict[Any, Any]:
        if not self.initial_chain or not self.follow_up_chain:
            raise ValueError("QA chains not initialized")

        docs = self.vector_store.similarity_search(question, k=3)

        if not self.conversation_history:
            answer = self.initial_chain.invoke(question)
        else:
            answer = self.follow_up_chain.invoke(question)

        return {
            "answer": answer,
            "source_documents": docs
        }


def main():
    parser = argparse.ArgumentParser(description='Cybersecurity RAG System')
    parser.add_argument('--vector_store_path', type=str, help='Path to save/load FAISS vector store')
    parser.add_argument('--use_openai_embeddings', action='store_true', help='Use OpenAI embeddings instead of HuggingFace')
    args = parser.parse_args()

    try:
        rag = CyberSecurityRAG(use_openai_embeddings=args.use_openai_embeddings)
        rag.initialize_embeddings(OPENAI_API_KEY)

        db = SessionLocal()
        try:
            documents = rag.load_db_data(db)
            if not documents:
                print("No CVE data found in the database.")
                return

            if args.vector_store_path and Path(args.vector_store_path).exists():
                print("Loading existing vector store...")
                if not rag.load_vector_store(args.vector_store_path):
                    print("Creating a new vector store...")
                    rag.create_vector_store(documents, args.vector_store_path)
            else:
                print("Creating a new vector store...")
                rag.create_vector_store(documents, args.vector_store_path)
        finally:
            db.close()

        rag.initialize_qa_chain(OPENAI_API_KEY)
        print("\nCybersecurity RAG System initialized. Enter your questions:")

        while True:
            question = input("\nQuestion: ").strip()
            if not question:
                continue

            result = rag.query(question)
            print("\nAnswer:", result["answer"])
            print("\nRelevant CVEs:")
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n--- Source {i} ---")
                print(doc.page_content)

    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")


if __name__ == "__main__":
    main()
