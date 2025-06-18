from neo4j import GraphDatabase

URI = "bolt+ssc://85759ac7.databases.neo4j.io"  # <- use +ssc for self-signed certs
username = "neo4j"
password = "Y4RrATrfzCwD1JXgXnyfBxX8T3MUW4OJtbsZypVhBjU"

with GraphDatabase.driver(URI, auth=(username, password)) as driver:
    driver.verify_connectivity()
    print("✅ Connected to Neo4j")
