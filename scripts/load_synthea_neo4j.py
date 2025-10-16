# scripts/load_synthea_neo4j.py
from pathlib import Path
import json
import pandas as pd
from neo4j_common import get_driver, ensure_synthea_constraints

# Point this to your Synthea FHIR output
FHIR_DIR = Path(r"C:\Users\yangm\Desktop\Mayo-Demo\output\fhir")

def _iter_resources():
    """Yield FHIR resources from all .json files (handles Bundle or single resource)."""
    for fp in FHIR_DIR.rglob("*.json"):
        try:
            with open(fp, encoding="utf-8") as f:
                doc = json.load(f)
        except Exception:
            continue
        if isinstance(doc, dict) and doc.get("resourceType") == "Bundle":
            for e in doc.get("entry", []):
                r = e.get("resource")
                if r:
                    yield r
        elif isinstance(doc, dict) and doc.get("resourceType"):
            yield doc
        elif isinstance(doc, list):
            for r in doc:
                if isinstance(r, dict) and r.get("resourceType"):
                    yield r

def _ref_id(ref):
    """FHIR reference like 'Patient/123' → '123'."""
    if not ref:
        return None
    return str(ref).split("/")[-1]

def run(database=None, limit=None):
    # Collect simple tabular rows for batching
    patients, encounters, observations = [], [], []

    for r in _iter_resources():
        rt = r.get("resourceType")
        if rt == "Patient":
            patients.append({
                "id": r.get("id"),
                "name": (r.get("name") or [{}])[0].get("text") or
                        " ".join((r.get("name") or [{}])[0].get("given", [])) + " " +
                        ((r.get("name") or [{}])[0].get("family") or ""),
                "birthDate": r.get("birthDate")
            })
        elif rt == "Encounter":
            encounters.append({
                "id": r.get("id"),
                "patient_id": _ref_id((r.get("subject") or {}).get("reference")),
                "start": (r.get("period") or {}).get("start"),
                "end":   (r.get("period") or {}).get("end"),
                "type":  ((r.get("type") or [{}])[0].get("text")) if r.get("type") else None
            })
        elif rt == "Observation":
            code = (((r.get("code") or {}).get("coding") or [{}])[0])
            valq = r.get("valueQuantity") or {}
            observations.append({
                "id": r.get("id"),
                "patient_id": _ref_id((r.get("subject") or {}).get("reference")),
                "encounter_id": _ref_id((r.get("encounter") or {}).get("reference")),
                "loinc": code.get("code"),
                "display": code.get("display"),
                "value": valq.get("value"),
                "unit": valq.get("unit"),
                "time": r.get("effectiveDateTime")
            })

        if limit and (len(patients)+len(encounters)+len(observations)) > limit:
            break

    # Convert to records for Neo4j
    p_rows = pd.DataFrame(patients).dropna(subset=["id"]).to_dict("records")
    e_rows = pd.DataFrame(encounters).dropna(subset=["id","patient_id"]).to_dict("records")
    o_rows = pd.DataFrame(observations).dropna(subset=["id"]).to_dict("records")

    driver = get_driver()
    with driver.session(database=database) as sess:
        ensure_synthea_constraints(sess)

        if p_rows:
            sess.run("""
            UNWIND $rows AS r
            MERGE (p:Patient {id: r.id})
            SET p.name = r.name, p.birthDate = r.birthDate
            """, rows=p_rows)

        if e_rows:
            sess.run("""
            UNWIND $rows AS r
            MERGE (e:Encounter {id: r.id})
            SET e.start = r.start, e.end = r.end, e.type = r.type
            WITH r, e
            MATCH (p:Patient {id: r.patient_id})
            MERGE (p)-[:HAD_ENCOUNTER]->(e)
            """, rows=e_rows)

        if o_rows:
            # Link to Encounter if available, else to Patient
            sess.run("""
            UNWIND $rows AS r
            MERGE (o:Observation {id: r.id})
            SET o.loinc = r.loinc, o.display = r.display,
                o.value = toFloat(r.value), o.unit = r.unit, o.time = r.time
            WITH r, o
            OPTIONAL MATCH (e:Encounter {id: r.encounter_id})
            WITH r, o, e
            FOREACH (_ IN CASE WHEN e IS NOT NULL THEN [1] ELSE [] END |
                MERGE (e)-[:HAS_OBSERVATION]->(o)
            )
            WITH r, o, e
            OPTIONAL MATCH (p:Patient {id: r.patient_id})
            WITH r, o, e, p
            FOREACH (_ IN CASE WHEN e IS NULL AND p IS NOT NULL THEN [1] ELSE [] END |
                MERGE (p)-[:HAS_OBSERVATION]->(o)
            )
            """, rows=o_rows)

    driver.close()
    print(f"✅ Loaded Synthea FHIR into Neo4j ({len(p_rows)} patients, {len(e_rows)} encounters, {len(o_rows)} observations).")

if __name__ == "__main__":
    run()
