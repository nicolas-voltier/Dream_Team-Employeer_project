from DB_neo4j import driver, _execute_query
from llama_index.core.graph_stores.types import EntityNode, Relation
from openai import AsyncOpenAI
import asyncio


class GraphProcessor:
    """
    A class to handle graph processing operations including querying, 
    document retrieval, and similarity computations.
    """
    
    def __init__(self):
        self.driver = driver
    
    async def embed_input(self, input: str, model: str):
        """Generate embeddings for input text using OpenAI API."""
        client = AsyncOpenAI()
        embedding_answer = await client.embeddings.create(input=input, model=model)
        return embedding_answer.data[0].embedding

    async def query_graph(self, question: str, threshold: float = 0.6, limit: int = 5, doc_limit: int | None = None, print_out=False):
        """Query the graph database for facts similar to the given question."""
        question_embedding = await self.embed_input(question, model="text-embedding-3-large")

        # Optional doc cap (backward-compatible: no cap if None)
        doc_limit_clause = ""
        if doc_limit is not None:
            doc_limit_clause = f"\nORDER BY max_similarity DESC\nLIMIT {doc_limit}\n"

        query = f"""
        MATCH (n:FACT)-[r]-(p:PAGE)-[]-(d:DOCUMENT)
        WHERE n.embedding IS NOT NULL
        WITH n, p, d,
            vector.similarity.cosine(n.embedding, {question_embedding}) AS similarity_fact,
            vector.similarity.cosine(r.embedding, {question_embedding}) AS similarity_page
        WITH n, p, d, similarity_fact, similarity_page, (similarity_fact + similarity_page)/2 AS similarity
        WHERE similarity > {threshold}
        ORDER BY similarity DESC // enforce ORDER BY similarity before top-k slice in query_graph
        WITH d,
            [fact IN collect({{fact: n.name, page: p.name, similarity: similarity}}) | fact][..{limit}] AS relevant_facts,
            max(similarity) AS max_similarity
        {doc_limit_clause}
        RETURN d.name AS document_name, d.summary AS document_summary, relevant_facts, max_similarity
        """

        graph_out = _execute_query(self.driver, query)

        formatted_output = ""
        for item in graph_out:
            formatted_output += f"From document: {item['document_name']}\n"
            formatted_output += f"With following executive summary: {item['document_summary']}\n"
            formatted_output += "Relevant facts found:\n"
            sorted_facts = sorted(item["relevant_facts"], key=lambda x: float(x['similarity']), reverse=True)
            for fact in sorted_facts:
                formatted_output += f"  - Fact: {fact['fact']}\n"
                formatted_output += f"  - From page: {fact['page']}\n"
                formatted_output += f"  - Similarity: {fact['similarity']:.3f}\n"
                formatted_output += "  --\n"
            formatted_output += "================================\n"

        if print_out:
            print(formatted_output)

        return formatted_output



    def find_corpus_labels(self, corpus_label: str = None):
        """Find all available corpus labels in the database."""
        find_corpus_query = """MATCH(c:CORPUS)
            return DISTINCT c.name as corpus_name"""
        corpus_list = _execute_query(self.driver, find_corpus_query)
        output_str = "The available corpus labels are: "
        for corpus in corpus_list:
            output_str += f"{corpus['corpus_name']}, "
        return output_str, corpus_list
        
    def get_document_with_descriptions(self, corpus_label: str = None):
        """Retrieve documents with descriptions, optionally filtered by corpus label."""
        if corpus_label is None:
            corpus_filter = ""
        else:
            _, corpus_list = self.find_corpus_labels(corpus_label)
            corpus_names = [corpus['corpus_name'] for corpus in corpus_list]
            if corpus_label in corpus_names:
                corpus_filter = f"WHERE c.name = '{corpus_label}'"
            else:
                return f"Corpus label {corpus_label} not found, the available labels are: {corpus_names}"

        query = f"""
        MATCH (d:DOCUMENT)-[]-(c:CORPUS)
        {corpus_filter}
        RETURN d.name AS document_name, d.summary AS document_summary, c.name AS bank_name
        """
        graph_out = _execute_query(self.driver, query)
        # for item in graph_out:
        #     print("From corpus under label: ", item["bank_name"])
        #     print("Document: ", item["document_name"])
        #     print("Summary: ", item["document_summary"])
        #     print("================================")
        return graph_out

    def compute_similarities(self):
        """Compute and store similarity relationships between facts."""
        query = """
        MATCH (n1:FACT), (n2:FACT)
        WHERE n1.embedding IS NOT NULL 
        AND n2.embedding IS NOT NULL 
        AND id(n1) < id(n2)
        WITH n1, n2, 
             vector.similarity.cosine(n1.embedding, n2.embedding) AS similarity
        WHERE similarity > 0.7
        MERGE (n1)-[r:SIMILARITY]->(n2)
        SET r.cosine_similarity = similarity
        """
        return _execute_query(self.driver, query)


# Create a default instance for backward compatibility


    def find_existing_node(self,node_name:str,label:str):
        check_exists_query=f"""
        MATCH(n:{label})
        WHERE n.name='{node_name}'
        RETURN n.name
        """
        found_nodes=_execute_query(self.driver,check_exists_query)
        return found_nodes


# if __name__ == "__main__":
#     graph_processor = GraphProcessor()
#     asyncio.run(graph_processor.query_graph("Did Malta come up in discussions within the HSBC corpus?",print_out=True))

if __name__ == "__main__":
    graph_processor = GraphProcessor()
    asyncio.run(
        graph_processor.query_graph(
            question="What was the dividend declared?",
            limit=5,
            print_out=True
        )
    )




