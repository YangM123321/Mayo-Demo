from py2neo import Graph, Node, Relationship

graph = Graph("bolt://localhost:7687", auth=("neo4j","testpass"))
graph.run("MATCH (n) DETACH DELETE n")  # dev only

diagnoses = [
    {"code":"250.00","name":"Type 2 Diabetes"},
    {"code":"285.9","name":"Anemia"},
]
labs = [
    {"loinc":"2345-7","name":"Glucose [Mass/volume] in Serum or Plasma"},
    {"loinc":"718-7","name":"Hemoglobin [Mass/volume] in Blood"},
]
links = [
    ("2345-7","250.00","INDICATES"),
    ("718-7","285.9","ASSOCIATED_WITH"),
]

tx = graph.begin()
for d in diagnoses:
    tx.merge(Node("Diagnosis", code=d["code"], name=d["name"]), "Diagnosis", "code")
for l in labs:
    tx.merge(Node("Lab", loinc=l["loinc"], name=l["name"]), "Lab", "loinc")
for loinc, icd, rel in links:
    lab = tx.evaluate("MATCH (l:Lab {loinc:$loinc}) RETURN l", loinc=loinc)
    dx  = tx.evaluate("MATCH (d:Diagnosis {code:$code}) RETURN d", code=icd)
    tx.create(Relationship(lab, rel, dx))
tx.commit()
print("KG loaded.")
