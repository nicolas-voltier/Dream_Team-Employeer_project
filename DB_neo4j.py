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


def get_graph_store():
    # Debug environment variables
    print(f"NEO4J_URI: {NEO4J_URI}")
    print(f"NEO4J_USERNAME: {NEO4J_USERNAME}")
    print(f"NEO4J_PASSWORD: {'*' * len(NEO4J_PASSWORD) if NEO4J_PASSWORD else 'None'}")
    
    if not NEO4J_URI or not NEO4J_PASSWORD:
        raise ValueError("NEO4J_URI and NEO4J_PASSWORD must be set in environment variables")
    
    try:
        # Initialize with connection resilience settings for cloud Neo4j
        graph_store = Neo4jPropertyGraphStore(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database="neo4j",
            refresh_schema=False,  
            timeout=60,  
            max_connection_lifetime=3600, 
            max_connection_pool_size=50,
            connection_acquisition_timeout=60
        )
        print("✅ Successfully connected to Neo4j with resilient settings")
        
        # Test the connection with a simple query
        try:
            test_result = graph_store.structured_query("RETURN 1 as test")
            print("✅ Connection test successful")
        except Exception as test_error:
            print(f"⚠️ Connection test failed: {test_error}")
        
        # Try schema refresh with timeout handling
        try:
            graph_store.refresh_schema()
            print("✅ Schema refreshed successfully")
        except Exception as schema_error:
            print(f"⚠️ Schema refresh failed (continuing anyway): {schema_error}")
        
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        raise
    
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
