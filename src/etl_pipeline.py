import pandas as pd
from py2neo import Graph

REQUIRED = {"patient_id","encounter_id","loinc","lab_value","unit","collected_date"}

def fetch_dx_for_loinc(graph, loinc):
    q = "MATCH (:Lab {loinc:$loinc})-[]->(d:Diagnosis) RETURN collect(d.code) AS dx"
    res = graph.run(q, loinc=loinc).data()
    return res[0]["dx"] if res else []

def main():
    df = pd.read_parquet("out/labs_clean.parquet")
    assert REQUIRED.issubset(df.columns), f"Schema mismatch: {set(df.columns)}"

    graph = Graph("bolt://localhost:7687", auth=("neo4j","testpass"))

    # attach dx codes from KG
    df["dx_codes"] = df["loinc"].apply(lambda x: fetch_dx_for_loinc(graph, x))

    # simple validity check
    df["is_value_valid"] = df["lab_value"].apply(lambda v: bool(pd.notna(v) and v > 0))

    df.to_parquet("out/labs_curated.parquet", index=False)
    print("Wrote out/labs_curated.parquet")

if __name__ == "__main__":
    main()
