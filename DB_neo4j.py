import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.core.graph_stores.types import (EntityNode,Relation)


working_directory=os.getcwd()
path_to_env=os.path.join(working_directory,".env")
load_dotenv(path_to_env)


NEO4J_URI=os.getenv("NEO4J_URI")    
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")


graph_store = None

def get_graph_store():
    global graph_store
    if graph_store is None:
        graph_store = Neo4jPropertyGraphStore(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
    return graph_store

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD),database="neo4j")

def _format_results(results):
    # Convert Neo4j results to a readable format
    formatted = []
    for record in results:
        formatted.append("\n".join(f"{k}: {v}" for k, v in record.items()))
    return "\n\n".join(formatted)

def _execute_query(driver, cypher,readable_format=False):
    with driver.session() as session:
        result = session.run(cypher)
        if readable_format:
            return _format_results(result)
        else:
            return [record.data() for record in result]