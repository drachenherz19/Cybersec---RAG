import json
import os


def load_mitre_data():
    file_path = os.path.join(os.path.dirname(__file__), '../data/mitre/enterprise-attack/enterprise-attack-1.0.json')
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None


def search_cve_in_mitre(cve_id):
    mitre_data = load_mitre_data()
    if not mitre_data:
        return []

    matched_entries = []
    for entry in mitre_data.get('objects', []):
        found = False
        if 'description' in entry and cve_id in entry['description']:
            found = True
        if 'external_references' in entry:
            for ref in entry['external_references']:
                if ref.get('description') and cve_id in ref['description']:
                    found = True
                if ref.get('external_id') and cve_id in ref['external_id']:
                    found = True

        if found:
            matched_entries.append({
                    'id': entry.get('id'),
                    'name': entry.get('name'),
                    'type': entry.get('type'),
                    'description': entry.get('description', 'No description available'),
                    'references': entry.get('external_references', [])
                    })

    return matched_entries
