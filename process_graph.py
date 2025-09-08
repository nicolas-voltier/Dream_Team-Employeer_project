from DB_neo4j import driver,_execute_query
from llama_index.core.graph_stores.types import EntityNode,Relation
from openai import AsyncOpenAI
import asyncio

async def embed_input(input:str,model:str):
    client = AsyncOpenAI()
    embedding_answer = await client.embeddings.create(input=input, model=model)
    return embedding_answer.data[0].embedding


async def query_graph(question:str, threshold:float=0.6,limit:int =5):
    question_embedding = await embed_input(question,model="text-embedding-3-large")
    query = f"""
    MATCH (n:FACT)-[]-(p:PAGE)-[]-(d:DOCUMENT)
    WHERE n.embedding IS NOT NULL
    WITH n, p, d, 
         reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * {question_embedding}[i]) /
         (sqrt(reduce(norm1 = 0.0, x IN n.embedding | norm1 + x * x)) * 
          sqrt(reduce(norm2 = 0.0, x IN {question_embedding} | norm2 + x * x))) AS similarity
    WHERE similarity > {threshold}
    WITH d, collect({{fact: n.name, page: p.name, similarity: similarity}}) AS relevant_facts, max(similarity) AS max_similarity
    RETURN d.name AS document_name, d.summary AS document_summary, relevant_facts
    ORDER BY max_similarity DESC
    LIMIT {limit}
    """

    graph_out= _execute_query(driver,query)

    for item in graph_out:
        print("From document: ",item["document_name"])
        print("With following executive summary: ",item["document_summary"])
        print("Relevant facts found:")
        for fact in item["relevant_facts"]:
            print(f"  - Fact: {fact['fact']}")
            print(f"  - From page: {fact['page']}")
            print(f"  - Similarity: {fact['similarity']:.3f}")
            print("  --")
        print("================================")
        
        


def compute_similarities():
    query = """
    MATCH (n1:FACT), (n2:FACT)
    WHERE n1.embedding IS NOT NULL 
    AND n2.embedding IS NOT NULL 
    AND id(n1) < id(n2)
    WITH n1, n2, 
         reduce(dot = 0.0, i IN range(0, size(n1.embedding)-1) | dot + n1.embedding[i] * n2.embedding[i]) /
         (sqrt(reduce(norm1 = 0.0, x IN n1.embedding | norm1 + x * x)) * 
          sqrt(reduce(norm2 = 0.0, x IN n2.embedding | norm2 + x * x))) AS similarity
    WHERE similarity > 0.7
    MERGE (n1)-[r:SIMILARITY]->(n2)
    SET r.cosine_similarity = similarity
    """
    return _execute_query(driver,query)


asyncio.run(query_graph("Was there any concern raised about Malta?"))





