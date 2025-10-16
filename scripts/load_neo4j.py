
# scripts/load_neo4j.py
import argparse, importlib, os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["synthea","mimic"], default=os.getenv("DATA_SOURCE","mimic"))
    p.add_argument("--database", default=os.getenv("NEO4J_DB"))
    args = p.parse_args()

    mod = "load_synthea_neo4j" if args.source == "synthea" else "load_mimic_neo4j"
    loader = importlib.import_module(mod)
    if hasattr(loader, "run"):
        loader.run(database=args.database)
    else:
        loader.main()

if __name__ == "__main__":
    main()
