# scripts/load_mimic_neo4j.py
from pathlib import Path
import pandas as pd
from neo4j_common import get_driver, ensure_mimic_constraints

BASE = Path(r"C:\Users\yangm\Desktop\Mayo-Demo\data\physionet.org\files\mimic-iv-ed-demo\2.2\ed")

def run(database=None):
    vitals = pd.read_csv(BASE / "vitalsign.csv.gz", compression="gzip")
    triage = pd.read_csv(BASE / "triage.csv.gz",    compression="gzip")

    # Keep it snappy for demos; remove .head(...) for full load
    vitals = vitals.head(5000)
    triage = triage.head(2000)

    driver = get_driver()
    with driver.session(database=database) as session:
        ensure_mimic_constraints(session)

        # Merge Patients + ED stays
        pairs = (pd.concat([vitals[['subject_id','stay_id']],
                            triage[['subject_id','stay_id']]])
                 .drop_duplicates()
                 .to_dict('records'))

        session.run("""
        UNWIND $rows AS r
        MERGE (p:Patient {subject_id: toInteger(r.subject_id)})
        MERGE (e:EDStay  {stay_id:   toInteger(r.stay_id)})
        MERGE (p)-[:HAD_ED_STAY]->(e)
        """, rows=pairs)

        # Attach vitals
        session.run("""
        UNWIND $rows AS r
        MATCH (e:EDStay {stay_id: toInteger(r.stay_id)})
        CREATE (v:Vital {
          charttime: r.charttime,
          temperature: toFloat(r.temperature),
          heartrate: toFloat(r.heartrate),
          resprate: toFloat(r.resprate),
          o2sat: toFloat(r.o2sat),
          sbp: toFloat(r.sbp),
          dbp: toFloat(r.dbp)
        })
        MERGE (e)-[:HAS_VITAL]->(v)
        """, rows=vitals.to_dict('records'))

        # Attach chief complaints
        session.run("""
        UNWIND $rows AS r
        MATCH (e:EDStay {stay_id: toInteger(r.stay_id)})
        WITH e, r WHERE r.chiefcomplaint IS NOT NULL AND r.chiefcomplaint <> ''
        MERGE (c:Complaint {text: r.chiefcomplaint})
        MERGE (e)-[:HAS_COMPLAINT]->(c)
        """, rows=triage.to_dict('records'))

    driver.close()
    print("✅ Loaded MIMIC-ED into Neo4j.")
