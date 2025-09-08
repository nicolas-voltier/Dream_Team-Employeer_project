from DB_neo4j import graph_store,driver,_execute_query
from llama_index.core.graph_stores.types import EntityNode,Relation
from pdf_processor import gen
from openai import AsyncOpenAI

async def embed_input(input:str,model:str):
    client = AsyncOpenAI()
    embedding_answer = await client.embeddings.create(input=input, model=model)
    return embedding_answer.data[0].embedding


def query_graph(question:str,limit:int, threshold:float=0.7):
    question_embedding = embed_input(question,model="text-embedding-3-large")
    query = f"""
    MATCH (n:FACT)
    WHERE n.embedding IS NOT NULL
    AND cosineSimilarity(n.embedding, {question_embedding}) > {threshold}
    RETURN n LIMIT {limit}
    """

    return _execute_query(driver,query)


def compute_similarities():
    query = """
    MATCH (n1:FACT), (n2:FACT)
    WHERE n1.embedding IS NOT NULL 
    AND n2.embedding IS NOT NULL 
    AND id(n1) < id(n2)
    WITH n1, n2, gds.similarity.cosine(n1.embedding, n2.embedding) AS similarity
    WHERE similarity > 0.7
    MERGE (n1)-[r:SIMILARITY]->(n2)
    SET r.cosine_similarity = similarity
    """



    return _execute_query(driver,query)






