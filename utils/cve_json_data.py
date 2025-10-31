from fastapi import HTTPException
from sqlalchemy.orm import Session
from itertools import combinations

from apis.nvd import fetch_cve_data
from model_examples.openai_script_old import generate_keywords
from db.models import CVE
from db.connection import SessionLocal


def save_cves_to_db(db: Session, new_vulnerabilities: list) -> int:
    """
    Save unique CVEs to the database.

    Args:
        db (Session): SQLAlchemy session.
        new_vulnerabilities (list): List of new CVEs to save.

    Returns:
        int: Number of CVEs saved to the database.
    """
    existing_cves = {cve.cve_id for cve in db.query(CVE.cve_id).all()}
    new_entries = []

    for v in new_vulnerabilities:
        cve_id = v["cve"]["id"]
        if cve_id in existing_cves:
            continue

        description = (
                next((desc["value"] for desc in v["cve"].get("descriptions", []) if desc["lang"] == "en"), "")
        )
        cvss_score = (
                v["cve"]["metrics"]
                .get("cvssMetricV31", [{}])[0]
                .get("cvssData", {})
                .get("baseScore")
        )
        published_date = v["cve"].get("published")
        last_modified_date = v["cve"].get("lastModified")

        additional_metadata = v

        new_entries.append(
                CVE(
                        cve_id=cve_id,
                        description=description,
                        cvss_score=cvss_score,
                        published_date=published_date,
                        last_modified_date=last_modified_date,
                        additional_metadata=additional_metadata,
                        )
                )

    db.add_all(new_entries)
    db.commit()
    return len(new_entries)


def process_user_query(user_query: str) -> dict:
    """
    Process a user query to extract keywords, search for CVEs in the database, and fallback to NVD API if needed.

    Args:
        user_query (str): The query input from the user.

    Returns:
        dict: A dictionary containing the processed results, including keywords and total CVEs.

    Raises:
        HTTPException: If no relevant keywords or CVEs are found, or if an internal error occurs.
    """
    try:
        # Step 1: Generate keywords using OpenAI
        keywords = generate_keywords(user_query)
        if not keywords:
            raise HTTPException(status_code=400, detail="No relevant keywords found")

        print("KEYWORDS for query:", keywords)

        # Step 2: Search Database First
        db = SessionLocal()
        cve_results = []
        try:
            # Search database for matching keywords
            for keyword in keywords.split("+"):
                keyword = keyword.strip()
                db_results = db.query(CVE).filter(CVE.description.ilike(f"%{keyword}%")).all()
                cve_results.extend(db_results)

            cve_results = list({cve.cve_id: cve for cve in cve_results}.values())  # Remove duplicates by CVE ID
        finally:
            db.close()

        if cve_results:
            return {
                    "user_query": user_query,
                    "generated_keywords": keywords,
                    "total_cves_found": len(cve_results),
                    "cves": [{"id": cve.cve_id, "description": cve.description} for cve in cve_results],
                    "message": "Results retrieved from the database.",
                    }

        # Step 3: Fallback to NVD API
        print("No results in database. Fetching from NVD API...")
        accumulated_vulnerabilities = []
        keyword_list = [k.strip() for k in keywords.split("+") if k.strip()]
        min_keywords = 1
        max_keywords = 3

        for r in range(max_keywords, min_keywords - 1, -1):
            for combo in combinations(keyword_list, r):
                query = " ".join(combo).strip()
                if not query:
                    continue

                search_results = fetch_cve_data(query=query)
                if search_results and "vulnerabilities" in search_results:
                    accumulated_vulnerabilities.extend(search_results.get("vulnerabilities", []))

        # Save new CVEs to the database
        if accumulated_vulnerabilities:
            db = SessionLocal()
            try:
                total_saved = save_cves_to_db(db, accumulated_vulnerabilities)
            finally:
                db.close()

            return {
                    "user_query": user_query,
                    "generated_keywords": keywords,
                    "total_cves_found": len(accumulated_vulnerabilities),
                    "total_cves_saved_to_db": total_saved,
                    "message": f"Fetched from NVD API and saved {total_saved} CVEs to the database.",
                    }

        raise HTTPException(status_code=404, detail="No CVE data found for any keyword combination")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
