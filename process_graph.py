from DB_neo4j import driver, _execute_query
from llama_index.core.graph_stores.types import EntityNode, Relation
from openai import AsyncOpenAI
from documentation_model import Fact
import asyncio

class Retrieved_fact():
    def __init__(self,page_id:str,page_name:str,fact_id:str,fact:str,question:str,similarity:float):
        self.page_id:str=page_id
        self.page_name:str=page_name
        self.fact_id:str=fact_id
        self.fact:str=fact
        self.question:str=question
        self.similarity:float=similarity
        self.retained:bool
    
    def retain(self):
        self.retained=True



class Retrieval_by_document:
    def __init__(self,document_summary:str,relevant_facts:list[Retrieved_fact],max_similarity:float,document_name:str,corpus_name:str):
        self.relevant_facts:list[Retrieved_fact]=relevant_facts
        self.max_similarity:float=max_similarity
        self.document_name:str=document_name
        self.document_summary:str=document_summary
        self.corpus_name:str=corpus_name

    def doc_level_outcome(self):
        """Generate formatted output string for the retrieval outcome."""
        sorted_facts = sorted(self.relevant_facts, key=lambda x: float(x.similarity), reverse=True)
        formatted_output = ""
        formatted_output += f"From document: {self.document_name}\n"
        formatted_output += f"With following executive summary: {self.document_summary}\n"
        formatted_output += "Relevant facts found:\n"
        
        for fact in sorted_facts:
            formatted_output += f"  - Fact (id: {fact.fact_id}): {fact.fact}\n"
            formatted_output += f"  - From page: {fact.page_name.split('_')[-1]}\n"
            formatted_output += f"  - Similarity: {fact.similarity:.3f}\n"
            formatted_output += "  --\n"
        formatted_output += "================================\n"
        return formatted_output

class Retrieval_overall:
    def __init__(self,retrieval_by_documents:list[Retrieval_by_document]):
        self.retrieval_by_documents:list[Retrieval_by_document]=retrieval_by_documents

    def print_outcome(self):
        formatted_output = ""
        for retrieval_by_document in self.retrieval_by_documents:
            formatted_output += retrieval_by_document.doc_level_outcome()
        return formatted_output

    def select_facts(self,fact_ids:list[str]):

        if len(fact_ids) == 0:
            return ""

        str_out="References:\n"
 
            
        for doc_ret in self.retrieval_by_documents:
            str_out+=f"  o {doc_ret.corpus_name} - {doc_ret.document_name}: page n°"
            fact_pages=[]
            for fact in doc_ret.relevant_facts:
                if fact.fact_id in fact_ids:
                    fact.retain()
                    fact_ids.remove(fact.fact_id)
                    fact_pages.append(fact.page_name.split("_")[-1])
            fact_pages=list(set(fact_pages))
            str_out+=f"{', page n°'.join(fact_pages)}\n"        

        if len(fact_ids) > 0:
            print(f"Warning: the following fact ids were not found: {fact_ids} check the fact ids")
        else:
            print(f"All facts were found and retained")
        return str_out

    def persist_answer(self,answer:str, user_question:str):
        answer_as_fact=Fact(answer=answer,questions=[user_question])
        pages_to_link=[]
        for retrieval_by_doc in self.retrieval_by_documents:
            pages_to_link.extend([r.page_id for r in retrieval_by_doc.relevant_facts if r.retained])
        answer_as_fact.create_fact_node_with_multiple_pages(pages_to_link)
        
    
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

    async def query_graph(self, question: str, threshold: float = 0.6, 
    doc_limit: int | None = None, total_fact_limit: int | None = None, print_out=False,fact_label:str="FACT", CORPUS:dict=None):
        """Query the graph database for facts similar to the given question."""
        question_embedding = await self.embed_input(question, model="text-embedding-3-large")

        corpus_filter = ""
        if CORPUS is not None:
            corpus_match = "-[]-(c:CORPUS)"
            
            if "excluded" in CORPUS:
                excluded_list = [f"'{item}'" for item in CORPUS['excluded']]
                corpus_filter += f" AND NOT c.name IN [{', '.join(excluded_list)}]"
            if "included" in CORPUS:
                included_list = [f"'{item}'" for item in CORPUS['included']]
                corpus_filter += f" AND c.name IN [{', '.join(included_list)}]"
        else:
            corpus_match = ""



        # Optional doc cap (backward-compatible: no cap if None)    
        doc_limit_clause = ""
        if doc_limit is not None:
            doc_limit_clause = f"\nORDER BY max_similarity DESC\nLIMIT {doc_limit}\n"

        if total_fact_limit is not None:
            # Query with total fact limit across all documents
            query_list = [f"""
            MATCH (n:{fact_label})-[r]-(p:PAGE)-[]-(d:DOCUMENT){corpus_match}  
            WHERE n.embedding IS NOT NULL {corpus_filter}""",
            f"""
            WITH n, p, d, r, c,
                vector.similarity.cosine(n.embedding, {question_embedding}) AS similarity_fact,
                vector.similarity.cosine(r.embedding, {question_embedding}) AS similarity_page
            """,
            f"""
            WITH n, r, c, p, d, similarity_fact, similarity_page, (similarity_fact + similarity_page)/2 AS similarity
            WHERE similarity > {threshold}
            WITH {{fact: n.name, fact_id: elementId(n), page: p.name, page_id: elementId(p), similarity: similarity, question: r.question, document_name: d.name, corpus: c.name, document_summary: d.summary}} AS fact_with_meta
            ORDER BY fact_with_meta.similarity DESC
            LIMIT {total_fact_limit}
            WITH collect(fact_with_meta) AS all_facts
            UNWIND all_facts AS fact
            WITH fact.document_name AS document_name, fact.corpus AS corpus, fact.document_summary AS document_summary,
                 collect({{fact: fact.fact, fact_id: fact.fact_id, page: fact.page, page_id: fact.page_id, similarity: fact.similarity, question: fact.question}}) AS relevant_facts,
                 max(fact.similarity) AS max_similarity
            {doc_limit_clause}
            RETURN document_name, corpus, document_summary, relevant_facts, max_similarity
            """]
        else:
            # Original query with per-document limit
            query_list = [f"""
            MATCH (n:{fact_label})-[r]-(p:PAGE)-[]-(d:DOCUMENT){corpus_match}  
            WHERE n.embedding IS NOT NULL {corpus_filter}""",
            f"""
            WITH n, p, d, r, c,
                vector.similarity.cosine(n.embedding, {question_embedding}) AS similarity_fact,
                vector.similarity.cosine(r.embedding, {question_embedding}) AS similarity_page
            """,
            f"""
            WITH n, r,c, p, d, similarity_fact, similarity_page, (similarity_fact + similarity_page)/2 AS similarity
            WHERE similarity > {threshold}
            ORDER BY similarity DESC // enforce ORDER BY similarity before top-k slice in query_graph
            WITH d, c,
                [fact IN collect({{fact: n.name, fact_id: elementId(n), page: p.name, page_id: elementId(p), similarity: similarity, question: r.question}}) | fact] AS relevant_facts,
                max(similarity) AS max_similarity
            {doc_limit_clause}
            RETURN d.name AS document_name, c.name as corpus, d.summary AS document_summary, relevant_facts, max_similarity
            """]
        # print(query_list[0])
        # print(query_list[2])
        query=query_list[0]+query_list[1]+query_list[2]
        graph_out = _execute_query(self.driver, query)


        outcomes = Retrieval_overall([])
        relevant_facts=[]
        for item in graph_out:
            # Extract page_ids and fact_ids from relevant_facts
            relevant_facts.extend([Retrieved_fact(page_id=fact['page_id'],
                                                    page_name=fact['page'],
                                                    fact_id=fact['fact_id'],
                                                    fact=fact['fact'],
                                                    question=fact['question'],
                                                    similarity=fact['similarity']) 
                                                    for fact in item['relevant_facts']])

            
            # Create Retrieval_outcome object
            outcomes.retrieval_by_documents.append(Retrieval_by_document(               
                    document_summary=item['document_summary'],
                    relevant_facts=relevant_facts,
                    max_similarity=item['max_similarity'],
                    document_name=item['document_name'],
                    corpus_name= item['corpus']
                ))


        if print_out:
            print(outcomes.print_outcome())

        return outcomes



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
        AND elementId(n1) < elementId(n2)
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




