# scripts/neo4j_common.py
from neo4j import GraphDatabase
import os

def get_driver():
    uri  = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    pwd  = os.getenv("NEO4J_PASSWORD", "testpass")  # set to your real password if different
    return GraphDatabase.driver(uri, auth=(user, pwd))

# --- Constraints for the MIMIC-IV ED demo graph ---
def ensure_mimic_constraints(session):
    session.run("""
    CREATE CONSTRAINT mimic_patient IF NOT EXISTS
    FOR (p:Patient) REQUIRE p.subject_id IS UNIQUE
    """)
    session.run("""
    CREATE CONSTRAINT mimic_stay IF NOT EXISTS
    FOR (e:EDStay) REQUIRE e.stay_id IS UNIQUE
    """)
    session.run("""
    CREATE CONSTRAINT mimic_complaint IF NOT EXISTS
    FOR (c:Complaint) REQUIRE c.text IS UNIQUE
    """)

# --- Example constraints for a Synthea graph (keep if you use Synthea) ---
def ensure_synthea_constraints(session):
    session.run("""
    CREATE CONSTRAINT synthea_patient IF NOT EXISTS
    FOR (p:Patient) REQUIRE p.id IS UNIQUE
    """)
    session.run("""
    CREATE CONSTRAINT synthea_enc IF NOT EXISTS
    FOR (e:Encounter) REQUIRE e.id IS UNIQUE
    """)
    session.run("""
    CREATE CONSTRAINT synthea_obs IF NOT EXISTS
    FOR (o:Observation) REQUIRE o.id IS UNIQUE
    """)
