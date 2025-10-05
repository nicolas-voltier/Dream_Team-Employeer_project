"""
Documentation model classes for the knowledge graph system.
Contains Document, Page, Corpus, Fact, and Question classes.
"""

import io
import base64
import os
import asyncio
import json
import uuid
from typing import List, Optional, Literal
from pathlib import Path
from datetime import datetime
from functools import wraps

import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
import openai
import numpy as np

from DB_neo4j import get_graph_store, EntityNode, Relation
# Fact and Question classes are defined in this file


class Question:
    """Represents a question associated with a fact."""
    
    def __init__(self, question: str):
        self.question: str = question
        self.embedding: List[float] = None


class Fact:
    """Represents a fact with associated questions and embeddings."""
    
    def __init__(self, answer: str, questions: List[str]):
        self.answer = answer
        self.questions: List[Question] = [Question(question=question) for question in questions]
        self.embedding: List[float] = None
        self.neo4j_id: str = None
    
    async def create_fact_node_with_multiple_pages(self, destination_nodes_ids: List[str], model: str = "text-embedding-3-large"):
        """Create a fact node in the graph with connections to multiple pages."""
        if self.embedding is None:
            print("Generating fact embeddings")
            client = openai.AsyncOpenAI()
            try:
                await self._generate_fact_embedding(client, model)
            finally:
                await client.close()
        
        try:
            graph_store = get_graph_store()
            fact_node = EntityNode(
                name=self.answer,
                label="FACT",
                properties={
                    "embedding": self.embedding,
                    "origin": "PERSISTED",
                    "created_on": datetime.now()
                }
            )
            self.neo4j_id = fact_node.id
            question_relations = []
            
            for destination_node_id in destination_nodes_ids:
                question_relations.extend([
                    Relation(
                        label="QUESTION",
                        properties={
                            "embedding": question.embedding,
                            "question": question.question,
                            "origin": "PERSISTED"
                        },
                        source_id=self.neo4j_id,
                        target_id=destination_node_id
                    ) for question in self.questions
                ])
            
            graph_store.upsert_nodes([fact_node])
            graph_store.upsert_relations(question_relations)
            return {"success": True, "fact_id": self.neo4j_id, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e), "fact_id": None}
    
    async def _generate_fact_embedding(self, client: openai.AsyncOpenAI, model: str):
        """Generate embedding for the fact and its questions."""
        embedding_answer = await client.embeddings.create(input=self.answer, model=model)
        self.embedding = embedding_answer.data[0].embedding
        
        for question in self.questions:
            question.embedding = (await client.embeddings.create(
                input=question.question, 
                model=model
            )).data[0].embedding


def neo4j_retry_with_backoff(
    max_retries: int = 5,
    base_delay: float = 3.0,
    max_delay: float = 180.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying Neo4j operations with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + 0.5 * np.random.random())
                    
                    print(f"Neo4j operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        raise e
                    
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay *= (0.5 + 0.5 * np.random.random())
                    
                    print(f"Neo4j operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    print(f"Retrying in {delay:.2f} seconds...")
                    import time
                    time.sleep(delay)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class Page:
    """Represents a single page with text and image."""
    
    def __init__(self, text: str = "", image: Optional[Image.Image] = None):
        self.text = text
        self.image: Image.Image = image
        self.image_base64: str = None
        self.page_id: str = None
        self.embedding: list[float] = None
        self.image_url: str = None
        self.best_representation: Literal["image", "text", None] = None
        self.summary: str = None
        self.keywords: list[str] = []
        self.facts: list[Fact] = []
        self.neo4j_id: str = None

    async def embed_page_facts(self, model):
        """Generate embeddings for all facts in this page."""
        client = openai.AsyncOpenAI()
        try:
            # Generate all embeddings concurrently
            await asyncio.gather(*[fact._generate_fact_embedding(client, model) for fact in self.facts if fact.questions is not None])
        finally:
            await client.close()

    def __str__(self):
        return f"Page(text_length={len(self.text)}, has_image={self.image is not None})"
   
    def __repr__(self):
        return self.__str__()
    
    def load_image(self):
        """Load image from URL."""
        self.image = Image.open(self.image_url)


class Document:
    """Represents a document as a list of pages."""
    
    def __init__(self, pages: Optional[List[Page]] = None, **kwargs):
        self.pages = pages if pages is not None else []
        self.path: str = Path(kwargs.get("path", None))
        self.description: str = kwargs.get("description", None)
        self.name: str = kwargs.get("name", None)
        self.metadata: dict = kwargs.get("metadata", {})     
        self.embedding: list[float] = kwargs.get("embedding", None)
        self.summary: str = kwargs.get("summary", None)
        self.doc_id: str = kwargs.get("doc_id", None)
        self.created_at: datetime = datetime.now()
        self.updated_at: datetime = datetime.now()
        self.neo4j_id: str = None

    def get_all_text(self) -> str:
        """Get all text from all pages concatenated."""
        return "\n\n".join(page.text for page in self.pages)

    async def setup_from_path(self, image_container_path: Path, screenshot_dpi: int = 150):
        """
        Process PDF completely - extract text and take screenshots for all pages.
        Returns a Document with pages containing both text and images.
        """
        print(f"Processing PDF (1st pass) from path: {self.path}")
        self.name = self.path.name
        
        # First extract text and sets embeddings
        with open(self.path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                pdf_page = pdf_reader.pages[page_num]
                text = pdf_page.extract_text()
                page = Page(text=text)
                page.page_id = f"{self.name}_page_{page_num:03d}"  # Generate unique page ID
                self.pages.append(page)
        
        await self._embed_document()
        
        # Then add screenshots to each page
        with fitz.open(str(self.path)) as pdf_document:
            os.makedirs(image_container_path, exist_ok=True)
            for page_num in range(len(pdf_document)):
                try:
                    # Get the page
                    page = pdf_document[page_num]
                    # Render page to image
                    mat = fitz.Matrix(screenshot_dpi / 72, screenshot_dpi / 72)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")
                    
                    # Save image
                    image_filename = f"{self.name}_page_{page_num:03d}.png"
                    image_path = image_container_path / image_filename
                    with open(image_path, "wb") as img_file:
                        img_file.write(img_data)
                    
                    # Update page with image info
                    if page_num < len(self.pages):
                        self.pages[page_num].image_url = str(image_path)
                        self.pages[page_num].load_image()
                        
                except Exception as e:
                    print(f"Error processing page {page_num}: {e}")
                    continue

    async def _embed_document(self):
        """Generate embeddings for all facts in the document."""
        client = openai.AsyncOpenAI()
        try:
            all_facts = []
            for page in self.pages:
                if hasattr(page, 'facts') and page.facts:
                    all_facts.extend([fact for fact in page.facts if fact.questions is not None])
            
            if all_facts:
                # Generate all embeddings concurrently
                await asyncio.gather(*[fact._generate_fact_embedding(client, model="text-embedding-3-large") for fact in all_facts])
                print(f"Generated embeddings for {len(all_facts)} facts across {len(self.pages)} pages")
            else:
                print("No facts found to embed in document")
        finally:
            await client.close()


class Corpus:
    """Represents a collection of documents from a specific bank."""
    
    def __init__(self, documents: list[Document], bank_name: str):
        self.documents: list[Document] = documents
        self.bank_name: str = bank_name
        self.neo4j_id: str = None
        
    def __len__(self):
        return len(self.documents)

    @neo4j_retry_with_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def upload_to_neo4j(self):
        """Upload the corpus to Neo4j database."""
        graph_store = get_graph_store()
        CorpusNode = EntityNode(name=self.bank_name, label="CORPUS", properties={"bank_name": self.bank_name})
        self.neo4j_id = CorpusNode.id
        graph_store.upsert_nodes([CorpusNode])
        for document in self.documents:
            upload_doc_to_neo4j(self.neo4j_id, document)


@neo4j_retry_with_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
def upload_doc_to_neo4j(corpus_node_id: str, pdf_document: Document, fact_label: str = "FACT") -> None:
    """
    Upload a document to Neo4j with its pages and facts.
    
    Args:
        corpus_node_id: ID of the corpus node to link to
        pdf_document: Document object to upload
        fact_label: Label for fact nodes (default: "FACT")
    """
    graph_store = get_graph_store()
    
    # Create document node
    doc_node = EntityNode(
        name=pdf_document.name,
        label="DOCUMENT",
        properties={
            "summary": pdf_document.summary,
            "created_at": pdf_document.created_at,
            "updated_at": pdf_document.updated_at
        }
    )
    pdf_document.neo4j_id = doc_node.id
    
    # Create page nodes and relations
    page_nodes = []
    doc_page_relations = []
    
    for page in pdf_document.pages:
        page_node = EntityNode(
            name=page.page_id,
            label="PAGE",
            properties={
                "text": page.text,
                "summary": page.summary,
                "keywords": page.keywords,
                "embedding": page.embedding
            }
        )
        page.neo4j_id = page_node.id
        page_nodes.append(page_node)
        
        # Create relation between document and page
        doc_page_relation = Relation(
            label="CONTAINS",
            source_id=pdf_document.neo4j_id,
            target_id=page.neo4j_id
        )
        doc_page_relations.append(doc_page_relation)
    
    # Create fact nodes and relations
    fact_nodes = []
    page_fact_relations = []
    
    for page in pdf_document.pages:
        for fact in page.facts:
            if fact.questions:  # Only create nodes for facts with questions
                fact_node = EntityNode(
                    name=fact.answer,
                    label=fact_label,
                    properties={
                        "embedding": fact.embedding,
                        "origin": "EXTRACTED"
                    }
                )
                fact.neo4j_id = fact_node.id
                fact_nodes.append(fact_node)
                
                # Create relation between page and fact
                page_fact_relation = Relation(
                    label="CONTAINS",
                    source_id=page.neo4j_id,
                    target_id=fact.neo4j_id
                )
                page_fact_relations.append(page_fact_relation)
                
                # Create question relations
                for question in fact.questions:
                    question_relation = Relation(
                        label="QUESTION",
                        properties={
                            "embedding": question.embedding,
                            "question": question.question,
                            "origin": "EXTRACTED"
                        },
                        source_id=fact.neo4j_id,
                        target_id=page.neo4j_id
                    )
                    page_fact_relations.append(question_relation)
    
    # Create corpus-document relation
    corpus_doc_relation = Relation(
        label="CONTAINS",
        source_id=corpus_node_id,
        target_id=pdf_document.neo4j_id
    )
    
    # Upload all nodes and relations
    all_nodes = [doc_node] + page_nodes + fact_nodes
    all_relations = doc_page_relations + page_fact_relations + [corpus_doc_relation]
    
    graph_store.upsert_nodes(all_nodes)
    graph_store.upsert_relations(all_relations)


def average_normalize_and_format_embedding(embeddings: list):
    """
    Average multiple embedding vectors element-wise, normalize, and format for Neo4j storage.
    
    Args:
        embeddings: list of embedding vectors (each vector is a list/array of floats)
        
    Returns:
        list: normalized average embedding formatted for Neo4j
    """
    if not embeddings:
        return None
    
    # Convert to numpy arrays for easier computation
    embeddings_array = np.array(embeddings)
    
    # Calculate element-wise average
    avg_embedding = np.mean(embeddings_array, axis=0)
    
    # Normalize the average embedding
    norm = np.linalg.norm(avg_embedding)
    if norm > 0:
        normalized_embedding = avg_embedding / norm
    else:
        normalized_embedding = avg_embedding
    
    # Convert back to list for Neo4j storage
    return normalized_embedding.tolist()
