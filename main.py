from fastapi import Depends
from fastapi import FastAPI, HTTPException
from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from apis.mitre import search_cve_in_mitre
from apis.nvd import fetch_cve_data
from apis.otx import fetch_otx_data_by_cve
from config import OPENAI_API_KEY
from cybersec_rag_max import CyberSecurityRAG
from db.connection import SessionLocal
from db.models import CVE
from model_examples.openai_script_old import generate_keywords, fetch_data_from_api
from utils.cve_json_data import process_user_query

app = FastAPI()
# Initialize the RAG system
rag = CyberSecurityRAG(use_openai_embeddings=False)
# Initialize Embeddings and Vector Store
try:
    rag.initialize_embeddings(OPENAI_API_KEY)

    print("Loading CVEs from the database...")
    db = SessionLocal()
    try:
        documents = rag.load_db_data(db)
        if not documents:
            print("No CVE data found in the database.")
        else:
            print(f"Loaded {len(documents)} CVEs from the database.")
            rag.create_vector_store(documents)
            print("Vector store created successfully.")
    finally:
        db.close()

    # Initialize QA chains
    rag.initialize_qa_chain(OPENAI_API_KEY)
    print("QA chains initialized successfully.")

except Exception as e:
    print(f"Error during initialization: {e}")


def get_db():
    """Provide a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
async def root():
    return {"message": "Welcome to the CVE Information Retrieval API"}


@app.get("/nvd")
async def get_nvd_data(query: str = None, cve_id: str = None, cpe_name: str = None, cvss_v3_severity: str = None, start_index: int = 0):
    return fetch_cve_data(query, cve_id, cpe_name, cvss_v3_severity, start_index)


@app.get("/mitre")
async def get_mitre_data(cve_id: str):
    results = search_cve_in_mitre(cve_id)
    if not results:
        raise HTTPException(status_code=404, detail="No MITRE data found for the given CVE")
    return results


@app.get("/otx")
async def get_otx_data(cve_id: str):
    data = fetch_otx_data_by_cve(cve_id)
    if not data:
        raise HTTPException(status_code=404, detail="No OTX data found for the given CVE")
    return data


# OpenAI Keyword Generation Endpoint
@app.get("/openai-keywords")
async def get_openai_keywords(user_query: str):
    return {"keywords": generate_keywords(user_query)}


# OpenAI Data Fetch Endpoint
@app.get("/fetch-openai-data")
async def get_openai_data(user_query: str):
    return {"response": fetch_data_from_api(user_query)}


# New User Query to CVEs Endpoint
@app.get("/user-query-cves")
async def user_query_cves(user_query: str):
    """
    Endpoint to process a user query, fetch CVE data, and save the results to a JSON file.

    Args:
        user_query (str): The query input from the user.

    Returns:
        dict: A dictionary containing the processed results, including keywords, total CVEs, and file path.

    Raises:
        HTTPException: If the query processing fails at any stage.
    """
    try:
        # Call the utility function to process the user query
        result = process_user_query(user_query)
        return result
    except HTTPException as http_exc:
        # If an HTTPException is raised, propagate it to the client
        raise http_exc
    except Exception as e:
        # Handle unexpected errors and return an internal server error
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        await websocket.send_text("Welcome to the Cybersecurity RAG Chat System! Ask your question or type 'exit' to leave.")

        while True:
            try:
                # Receive the user query
                user_query = await websocket.receive_text()
                if user_query.lower() == "exit":
                    await websocket.send_text("Thank you for using the Cybersecurity RAG System. Goodbye!")
                    break

                print(f"User Query: {user_query}")

                # Query the RAG system
                try:
                    result = rag.query(user_query)  # Query the RAG system
                    answer = result["answer"]
                except Exception as e:
                    print(f"Error querying RAG system: {e}")
                    await websocket.send_text("An error occurred while processing your query. Please try again.")
                    continue

                # Send the response to the user
                await websocket.send_text(f"\nAnswer: {answer}")

                # Display related CVEs directly
                source_documents = result.get("source_documents", [])
                if source_documents:
                    await websocket.send_text("\nRelevant CVEs:")
                    for i, doc in enumerate(source_documents, 1):
                        content = doc.page_content.strip()
                        await websocket.send_text(f"\n--- Source {i} ---\n{content}")
                else:
                    await websocket.send_text("\nNo relevant CVEs found for this query.")

            except Exception as session_error:
                print(f"Error during interactive session: {session_error}")
                await websocket.send_text("An unexpected error occurred. Please try again.")
    except WebSocketDisconnect:
        print("Client disconnected")


@app.get("/get-db-cves")
def get_cves(db: Session = Depends(get_db), offset: int = 0):
    """
    Retrieve CVEs and their IDs from the database.

    Args:
        db (Session): SQLAlchemy database session.
        offset (int): Offset for pagination.
        limit (int): Number of records to fetch.

    Returns:
        JSON response with total CVEs and their IDs.
    """
    try:
        total_cves = db.query(CVE).count()  # Total CVEs count
        cve_ids = db.query(CVE.cve_id).offset(offset).all()  # Fetch paginated CVE IDs

        # Format response
        cve_ids_with_numbers = [
                {"number": index + 1 + offset, "cve_id": cve_id[0]} for index, cve_id in enumerate(cve_ids)
                ]

        return {"total": total_cves, "cves": cve_ids_with_numbers}
    except Exception as e:
        print(f"Error fetching CVEs: {e}")
        return {"error": "Failed to fetch CVEs from the database."}
