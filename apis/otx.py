import requests
from fastapi import HTTPException
from config import OTX_API_KEY


def fetch_otx_data_by_cve(cve_id):
    headers = {'X-OTX-API-KEY': OTX_API_KEY}
    api_url = f"https://otx.alienvault.com/api/v1/indicators/cve/{cve_id}/general"

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data from OTX: {str(e)}")
