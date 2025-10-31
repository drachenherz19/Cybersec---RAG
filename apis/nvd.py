import json
from datetime import datetime

import requests
from sqlalchemy.orm import Session

from db.connection import SessionLocal
from db.models import CVE


def load_existing_cves(file_path: str) -> dict:
    """
    Load existing CVEs from the JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Existing CVEs or an empty structure if the file does not exist.
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"totalResults": 0, "vulnerabilities": []}


def parse_date(date_str):
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return None


def save_cves_to_db(db: Session, new_vulnerabilities: list):
    existing_cves = {cve.cve_id for cve in db.query(CVE.cve_id).all()}

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
        published_date = parse_date(v["cve"]["published"])
        last_modified_date = parse_date(v["cve"]["lastModified"])

        additional_metadata = v

        # Insert into the database
        cve = CVE(
            cve_id=cve_id,
            description=description,
            cvss_score=cvss_score,
            published_date=published_date,
            last_modified_date=last_modified_date,
            additional_metadata=additional_metadata,
        )
        db.add(cve)

    db.commit()
    print(f"{len(new_vulnerabilities)} CVEs processed and saved to the database.")



def fetch_cve_data(query=None, cve_id=None, cpe_name=None, cvss_v3_severity=None, start_index=0):
    api_url = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    params = {'startIndex': start_index}

    if query:
        params['keywordSearch'] = query.strip()
    if cve_id:
        params['cveId'] = cve_id
    if cpe_name:
        params['cpeName'] = cpe_name
    if cvss_v3_severity:
        params['cvssV3Severity'] = cvss_v3_severity

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Query: {query}, Total CVEs: {data.get('totalResults')}")

        # Save unique CVEs to database
        if "vulnerabilities" in data:
            db = SessionLocal()
            try:
                save_cves_to_db(db, data["vulnerabilities"])
            finally:
                db.close()

        return data
    except requests.RequestException as e:
        print(f"Error fetching data from NVD for query '{query}': {e}")

    return None
