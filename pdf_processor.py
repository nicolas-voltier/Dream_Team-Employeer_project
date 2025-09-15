"""
Simple PDF processor with classes for document handling.
"""

import io
import base64
from typing import List, Optional, Literal
from pathlib import Path
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from datetime import datetime
import uuid
import openai
import json
import numpy as np
import os
import dotenv
import asyncio
import random
import time
from functools import wraps
from process_graph import GraphProcessor
graph_processor=GraphProcessor()

#Loading environment variables
dotenv.load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

from DB_neo4j import get_graph_store, EntityNode,Relation
graph_store=get_graph_store()
def async_retry_with_exponential_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for async functions to retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        jitter: Add random jitter to prevent thundering herd
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (
                    openai.APIConnectionError,
                    openai.APITimeoutError,
                    openai.InternalServerError,
                    openai.RateLimitError
                ) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        print(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    print(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}")
                    print(f"Retrying in {delay:.2f} seconds...")
                    
                    await asyncio.sleep(delay)
                except Exception as e:
                    # For non-retryable exceptions, raise immediately
                    print(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise e
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

def neo4j_retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 2.0,
    max_delay: float = 120.0
):
    """
    Decorator for Neo4j operations to retry on connection/timeout errors.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import neo4j
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except (
                    neo4j.exceptions.SessionExpired,
                    neo4j.exceptions.ServiceUnavailable,
                    neo4j.exceptions.TransientError,
                    TimeoutError
                ) as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        print(f"Max retries ({max_retries}) exceeded for {func.__name__}")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    delay = delay * (0.5 + random.random() * 0.5)  # Add jitter
                    
                    print(f"Neo4j operation failed (attempt {attempt + 1}): {str(e)}")
                    print(f"Retrying in {delay:.2f} seconds...")
                    
                    time.sleep(delay)
                except Exception as e:
                    # For non-retryable exceptions, raise immediately
                    print(f"Non-retryable Neo4j error in {func.__name__}: {str(e)}")
                    raise e
            
            raise last_exception
        
        return wrapper
    return decorator

def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')
   
@async_retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
async def _send_images_to_openai(images_base64: list[str], prompt: str, model: str) -> str:
    """Send image to OpenAI and return the response text."""
    client= openai.AsyncOpenAI()
    try:
        messages=[{"role": "system", "content": [{"type": "text", "text": prompt}]}]
        for image_base64 in images_base64:
            messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]})
        if model.startswith("gpt-5"):
            response = await client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=messages
            )

    finally:
        await client.close()
        
    return response.choices[0].message.content

@async_retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
async def _send_text_to_openai(text: str, prompt: str, model: str) -> str:
    """Send text to OpenAI and return the response text."""
    client= openai.AsyncOpenAI()
    try:
        messages=[{"role": "system", "content": [{"type": "text", "text": prompt}]}]
        messages.append({"role": "user", "content": [{"type": "text", "text": text}]})
        if model.startswith("gpt-5"):
            response = await client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=messages
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=messages
            )

    finally:
        await client.close()
        
    return response.choices[0].message.content

@async_retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
async def _generate_page_embedding(client: openai.AsyncOpenAI, page: 'Page', model: str):
    """Generate embedding for a single page."""
    if page.summary is not None:
        input_text = page.summary
    else:
        input_text = page.text
    embedding_response = await client.embeddings.create(input=input_text, model=model)
    page.embedding = embedding_response.data[0].embedding


@async_retry_with_exponential_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
async def _generate_fact_embedding(client: openai.AsyncOpenAI, fact: 'Fact', model: str):
    """Generate embedding for a single fact."""
    embedding_answer = await client.embeddings.create(input=fact.answer, model=model)
    
    fact.embedding = embedding_answer.data[0].embedding
    for question in fact.questions:
        question.embedding= (await client.embeddings.create(input=question.question, model=model)).data[0].embedding
    

async def _process_page_summary(page: 'Page', page_summary_prompt: str, openai_model: str) -> str:
    """Process a single page to generate summary and keywords."""
    page_image_base64 = _image_to_base64(page.image)
    page_summary = await _send_images_to_openai([page_image_base64], page_summary_prompt, openai_model)
    page_summary_json = json.loads(page_summary)
    page.summary = page_summary_json["summary"]
    page.keywords = page_summary_json["keywords"]
    return page_summary

async def _process_page_representation(page_idx: int, page: 'Page', total_pages: int, openai_model: str):
    """Process a single page to determine best representation."""
    try:
        print(f"Detecting best representation for page {page_idx}/{total_pages}")
        # Convert PIL Image to base64
        image_base64 = _image_to_base64(page.image)
        prompt = """
        You are analyzing the screenshot of a pdf page of a page and you must decide if the page has many positional information;
        Positional data are: table, list, timeline, graph, map, image, etc...
        Expected output: a json file with below structure 
        {
            "best_representation": "image" | "text" | None,
            "reasoning": "reasoning for the best representation"
        }
        """
        # Send to OpenAI
        response = await _send_images_to_openai([image_base64], prompt, openai_model)
        
        # Update page with AI response
        json_data = json.loads(response)
        page.best_representation = json_data["best_representation"]  # Mark as processed from image
        return json_data
        
    except Exception as e:
        raise

class Fact:
    def __init__(self, answer:str,questions:list[str]):
        self.answer=answer
        self.questions:list[Question]=[Question(question=question) for question in questions]
        self.embedding:list[float] = None
        self.neo4j_id:str=None


class Question:
    def __init__(self,question:str):
        self.question:str=question
        self.embedding:list[float] =  None



        
class Page:
    """Represents a single page with text and image."""
    
    def __init__(self, text: str = "", image: Optional[Image.Image] = None):
        self.text = text
        self.image: Image.Image = image
        self.image_base64:str = None
        self.page_id:str = None
        self.embedding:list[float] = None
        self.image_url:str = None
        self.best_representation: Literal["image", "text",None] = None
        self.summary:str = None
        self.keywords:list[str] = []
        self.facts:list[Fact] = []
        self.neo4j_id:str=None

    async def embed_page_facts(self,model):
        
        client = openai.AsyncOpenAI()
        try:
            # Generate all embeddings concurrently
            await asyncio.gather(*[_generate_fact_embedding(client, fact, model) for fact in self.facts if fact.questions is not None])
        finally:
            await client.close()

    def __str__(self):
        return f"Page(text_length={len(self.text)}, has_image={self.image is not None})"
   
    def __repr__(self):
        return self.__str__()  
    def load_image(self):
        self.image = Image.open(self.image_url)

class Document:
    """Represents a document as a list of pages."""
    
    def __init__(self, pages: Optional[List[Page]] = None, **kwargs):
    
        self.pages = pages if pages is not None else []
        self.path:str = Path(kwargs.get("path", None))
        self.description:str = kwargs.get("description", None)
        self.name:str = kwargs.get("name", None)
        self.metadata:dict = kwargs.get("metadata", {})     
        self.embedding:list[float] = kwargs.get("embedding", None)
        self.summary:str = kwargs.get("summary", None)
        self.doc_id:str = kwargs.get("doc_id", None)
        self.created_at:datetime = datetime.now()
        self.updated_at:datetime = datetime.now()
        self.neo4j_id:str=None

    
    def get_all_text(self) -> str:
        """Get all text from all pages concatenated."""
        return "\n\n".join(page.text for page in self.pages)


    async def setup_from_path(self,image_container_path:Path, screenshot_dpi: int = 150):
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

        with  fitz.open(str(self.path)) as pdf_document:
            os.makedirs(image_container_path, exist_ok=True)
            for page_num in range(len(pdf_document)):
                try:
                    # Get the page
                    page = pdf_document[page_num]
                    
                    # Create transformation matrix for desired DPI
                    mat = fitz.Matrix(screenshot_dpi / 72, screenshot_dpi / 72)
                    
                    # Render page to pixmap
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Convert to PIL Image
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Add image to the corresponding page and save to disk
                    if page_num < len(self.pages):
                        self.pages[page_num].image = img

                        
                        
                        # Create filename based on PDF name and page number
                        
                        image_filename = f"{self.name}_page_{page_num:03d}.png"
                        image_path = Path(image_container_path) / image_filename
                        
                        # Save the image
                        img.save(image_path)
                        # Set the image_url property
                        self.pages[page_num].image_url = str(image_path)
                        
                except Exception as e:
                    print(f"Warning: Could not process screenshot for page {page_num}: {e}")

    async def _embed_document(self, model: str = "text-embedding-3-large") -> list[float]:
        """
        Get OpenAI embeddings for a text.
        """
        print(f"Embedding document: {self.path}")
        client = openai.AsyncOpenAI()
        try:
            # Generate all embeddings concurrently
            await asyncio.gather(*[_generate_page_embedding(client, page, model) for page in self.pages])
        finally:
            await client.close()
        page_embeddings = [page.embedding for page in self.pages]
        self.embedding = average_normalize_and_format_embedding(page_embeddings)


    async def generate_document_summaries(self, openai_model: str = "gpt-5-nano") -> tuple[str, str]:
        print(f"Getting document summaries for document: {self.path}")
        document_summary_prompt = """
        <role>
        You are provided with all pages from a document.
        Your task is to analyze the document in order to provide a description of the content (max 50 words) of the document. 
        Maximize the meaningful words, avoid unnecessary words.
        </role>
        <context>
        The document is part of a corpus of documents. 
        {corpus}
        </context>
        <format>
        Expected output: a json object in the following format:
        {{
            "summary": "summary of the document",
        }}
        </format>
        
        """
        
        page_summary_prompt = """ 
        <role>
        You are provided with a a page from a document.
        Your task is to analyze the page and provide a summary of its content.
        </role>

        <context>
        The document is: {document}
        </context>
        <format>
        Expected output: a json object in the following format:
        {{
            "summary": "summary of the page",
            "keywords": "keywords describing the page"
        }}
        </format>
        """

        all_pages_base64 = []
        for page in self.pages:
            if page.image is None:
                page.load_image()
            
            page_image_base64 = _image_to_base64(page.image)
            all_pages_base64.append(page_image_base64)

        # Get document summary first
        document_summary = await _send_images_to_openai(all_pages_base64, document_summary_prompt.format(corpus=""), openai_model)
        doc_summar_json = json.loads(document_summary)
        self.summary = doc_summar_json["summary"]
        
        # Process all pages concurrently
        # Use asyncio.gather for concurrent processing
        page_summaries = await asyncio.gather(*[_process_page_summary(page, page_summary_prompt.format(document=document_summary), openai_model) for page in self.pages])
        
        return document_summary, page_summaries


    async def detect_best_representation(self,openai_model: str = "gpt-5-nano") -> None:
        """
        Send each page as an image to OpenAI with a custom prompt.
        Updates each page's text with the AI response.
        
        Args:
            prompt: The prompt to send with each image
            openai_model: The OpenAI model to use for analysis
        """

        print(f"Detecting best representation for document: {self.path}")
        
        # Validate all pages have images first
        for page_idx, page in enumerate(self.pages):
            if page.image is None:
                raise ValueError(f"Page {page_idx} has no image")
        
        # Process all pages concurrently
        await asyncio.gather(*[_process_page_representation(page_idx, page, len(self.pages), openai_model) for page_idx, page in enumerate(self.pages)])
    

    async def _generate_page_1stfacts(self,page:Page,context=None, focus= None,openai_model: str = "gpt-5-nano"):
        focus_prompt= f"Focus on {focus}" if focus is not None else ""
        prompt=f"""
        <task>
        You are provided a page from a document. Extract from this page all facts that a reader must know. 
        Each fact must be associated with a questions from a reader that this fact can answer.
        {focus_prompt}
        </task>
        <context>
        Description of document from which page is extracted:
        {context}
        </context>
        <format>
        Expected output: a list of json object corresponding to the list of facts extracted from the page.

        {{
            "facts":[ 
        {{ 'fact': 'fact from the page', 'questions': ['questions answered by this fact', '...'] }},
        {{ 'fact': 'fact from the page', 'questions': ['questions answered by this fact', '...'] }},
        ...
         ]
        }}
        </format>

        """
        if page.best_representation=="image":
            if page.image_base64 is None:
                if page.image == None:
                    page.load_image()
                page.image_base64 = _image_to_base64(page.image)
            response = await _send_images_to_openai([page.image_base64], prompt, openai_model)
        else:
            response = await _send_text_to_openai(page.text, prompt, openai_model)
        json_data = json.loads(response)
        print(json_data)
        page.facts=[]
        for f in json_data["facts"]:
            try:
                new_fact=Fact(answer=f["fact"],questions=f["questions"])
                page.facts.append(new_fact)
    
            except Exception as e:
                print(f"Error processing fact: {f}")
                print(e)
                continue
        return page.facts
    
    async def generate_document_1stfacts(self, context:str=None, focus:str=None,openai_mode="gpt-5-nano"):
        # Generate facts for all pages concurrently
        await asyncio.gather(*[
            self._generate_page_1stfacts(page=page, context=context, focus=focus, openai_model=openai_mode) 
            for page in self.pages
        ])

    async def embed_document_facts(self, model: str = "text-embedding-3-large"):
        """
        Generate embeddings for all facts across all pages in the document.
        """
        print(f"Embedding all facts for document: {self.path}")
        client = openai.AsyncOpenAI()
        try:
            # Collect all facts from all pages
            all_facts = []
            for page in self.pages:
                if hasattr(page, 'facts') and page.facts:
                    all_facts.extend([fact for fact in page.facts if fact.questions is not None])
            
            if all_facts:
                # Generate all embeddings concurrently
                await asyncio.gather(*[_generate_fact_embedding(client, fact, model) for fact in all_facts])
                print(f"Generated embeddings for {len(all_facts)} facts across {len(self.pages)} pages")
            else:
                print("No facts found to embed in document")
        finally:
            await client.close()




    def page_count(self) -> int:
        """Get the number of pages."""
        return len(self.pages)
    
    def __len__(self):
        return len(self.pages)
    
    def __getitem__(self, index):
        return self.pages[index]
    
    def __iter__(self):
        return iter(self.pages)
    
    def __str__(self):
        return f"Document({len(self.pages)} pages)"
    
    def __repr__(self):
        return self.__str__()

class Corpus:
    def __init__(self, documents:list[Document], bank_name:str):
        self.documents:list[Document]=documents
        self.bank_name:str=bank_name
        self.neo4j_id:str=None
        
    
    def __len__(self):
        return len(self.documents)

    @neo4j_retry_with_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
    def upload_to_neo4j(self):
        CorpusNode=EntityNode(name=self.bank_name,label="CORPUS",properties={"bank_name":self.bank_name})
        self.neo4j_id=CorpusNode.id
        graph_store.upsert_nodes([CorpusNode])
        for document in self.documents:
            upload_doc_to_neo4j(self.neo4j_id,document)


def  average_normalize_and_format_embedding(embeddings:list):
    """
    Average multiple embedding vectors element-wise, normalize, and format for Neo4j storage.
    
    Args:
        embeddings: list of embedding vectors (each vector is a list/array of floats)
        
    Returns:
        list: normalized average embedding formatted for Neo4j
    """
    if not embeddings:
        raise ValueError("No embeddings provided")
    
    # Convert to numpy array and take element-wise mean across vectors (axis=0)
    embeddings_array = np.array(embeddings)
    average_embedding = np.mean(embeddings_array, axis=0)
    
    # Normalize using L2 norm
    norm = np.linalg.norm(average_embedding)
    if norm > 0:
        normalized_embedding = average_embedding / norm
    else: 
        raise ValueError("Average embedding has zero norm")
    
    # Convert to list and ensure proper format for Neo4j
    return normalized_embedding.tolist()

@neo4j_retry_with_backoff(max_retries=5, base_delay=3.0, max_delay=180.0)
def upload_doc_to_neo4j(corpus_node_id:str,pdf_document:Document) -> None:
    """
    Upload the PDF to Neo4j.
    """
    print(f"Uploading PDF to Neo4j: {pdf_document.path}")

    
    node_to_insert=[]

    relation_to_insert=[]


    document_node=EntityNode(name=pdf_document.name,
                            label="DOCUMENT",
                            properties={"path":str(pdf_document.path),
                                        "description":pdf_document.description,
                                        "name":pdf_document.name,
                                        "metadata":json.dumps(pdf_document.metadata) if pdf_document.metadata else "{}",
                                        "embedding":pdf_document.embedding,
                                        "summary":pdf_document.summary,
                                        "doc_id":pdf_document.doc_id,
                                        "created_at":pdf_document.created_at.isoformat() if pdf_document.created_at else None,
                                        "updated_at":pdf_document.updated_at.isoformat() if pdf_document.updated_at else None})
    node_to_insert.append(document_node)
    pdf_document.neo4j_id=document_node.id

    document_corpus_relation=Relation(label="PART_OF",source_id=pdf_document.neo4j_id,target_id=corpus_node_id)
    relation_to_insert.append(document_corpus_relation)

    for page in pdf_document.pages:
        page_node=EntityNode(name=page.page_id,label="PAGE",properties={"text":page.text,
        #"image":page.image,
        "embedding":page.embedding,
        "summary":page.summary,
        "keywords":page.keywords,
        "image_url":page.image_url,
        "best_representation":page.best_representation})
        node_to_insert.append(page_node)
        page.neo4j_id=page_node.id
        
        relation_to_insert.append(Relation(label="CONTAINS",properties={"page_id":page.page_id},source_id=pdf_document.neo4j_id,target_id=page.neo4j_id))

        for fact in page.facts:
            fact_node=EntityNode(name=fact.answer,label="FACT",properties={"embedding": fact.embedding})
            node_to_insert.append(fact_node)
            fact.neo4j_id=fact_node.id
            for question in fact.questions:
                # Only create relation if question has proper embedding and question text
                if hasattr(question, 'embedding') and question.embedding is not None and hasattr(question, 'question'):
                    question_relation=Relation(label="QUESTION",properties={"embedding":question.embedding,"question":question.question},source_id=fact.neo4j_id,target_id=page.neo4j_id)
                    relation_to_insert.append(question_relation)
                else:
                    print(f"Warning: Skipping question relation - missing embedding or question text: {question}")

    print(f"Upserting {len(node_to_insert)} nodes to Neo4j")
    graph_store.upsert_nodes(node_to_insert)

    print(f"Upserting {len(relation_to_insert)} relations to Neo4j")
    graph_store.upsert_relations(relation_to_insert)


    # def download_from_neo4j(self):
    #     """
    #     Download the PDF from Neo4j.
    #     """
    #     from DB_neo4j import graph_store, EntityNode,Relation
    #     document_node=graph_store.get_node(self.pdf_document.doc_id)
    #     self.pdf_document.name = document_node.name
    #     self.pdf_document.url = document_node.properties["url"]
    #     self.pdf_document.description = document_node.properties["description"]
    #     self.pdf_document.metadata = json.loads(document_node.properties["metadata"]) if document_node.properties["metadata"] else {}
    #     self.pdf_document.embedding = document_node.properties["embedding"]
    #     self.pdf_document.doc_id = document_node.id
    #     self.pdf_document.created_at = datetime.fromisoformat(document_node.properties["created_at"]) if document_node.properties["created_at"] else None
    #     self.pdf_document.updated_at = datetime.fromisoformat(document_node.properties["updated_at"]) if document_node.properties["updated_at"] else None

def upsert_if_not_exist(node_name:str,label:str,properties:dict):
    """upsert if not exist reverts with node_id"""
    found_nodes=graph_processor.find_existing_node(node_name,label)
    if len(found_nodes)==1:
        return found_nodes[0].id
    elif len(found_nodes)==0:
        node_to_insert=EntityNode(name=node_name,label=label,properties=properties)
        graph_store.upsert_nodes([node_to_insert])
        print(f"Upserted node {node_name} with label {label} and properties {properties}")
        return node_to_insert.id
    else:
        
        raise ValueError(f"Found multiple nodes with name {node_name} and label {label}")


async def preproc_bank_documents(folder_path,file_list=None):  
    # qdocuments=[]
    corpus=Corpus(documents=[], bank_name=Path(folder_path).name)
    corpus.neo4j_id=upsert_if_not_exist(corpus.bank_name,"CORPUS",{"bank_name":corpus.bank_name})
    for file in os.listdir(folder_path):
        if len(graph_processor.find_existing_node(file,"DOCUMENT")) == 0:
           
            process=False
            if file in file_list:
                process=True
            elif file_list is None:
                process=True
            if process:
                print(f"------------- Processing file: {file} ---------------------")
                pdf_path=os.path.join(folder_path,file)

                document=Document(path=pdf_path)
                await document.setup_from_path(image_container_path=Path("images"))

                await document.generate_document_summaries()
                await document.detect_best_representation()
                await document.generate_document_1stfacts()

                # for page in document.pages:
                #     await page.embed_page_facts(model="text-embedding-3-large")
                await document.embed_document_facts(model="text-embedding-3-large")
                upload_doc_to_neo4j(corpus.neo4j_id,document)
        else:
            print(f"------------- File already exists: {file} ---------------------")


    corpus.upload_to_neo4j()


            #upload_to_neo4j(document)




if __name__ == "__main__":
    pdf_path=r'C:\Users\volti\OneDrive\Documents\Python_projects\Dream_Team-Employeer_project\data\Barclays'
    # for fpath in os.listdir(pdf_path):
    #     print(fpath)
    file_list=["Barclays_2022_Q1_ResultsQA_Transcript.pdf"
                "Barclays_2022_Q2_H1_ResultsQA_Transcript.pdf" ,
                "Barclays_2022_Q3_ResultsQA_Transcript.pdf" ,
                "Barclays_2022_Q4_ResultsQA_Transcript.pdf" ,
                "Barclays_2023_Q1_ResultsQA_Transcript.pdf" ,
                "Barclays_2023_Q2_ResultsQA_Transcript.pdf" ,
                "Barclays_2023_Q3_ResultsQA_Transcript.pdf" ,
                "Barclays_2024_Q1_ResultsQA_Transcript.pdf" ,
                "Barclays_2024_Q2_ResultsQA_Transcript.pdf" ,
                "Barclays_2024_Q3_ResultsQA_Transcript.pdf" ,
                "Barclays_2024_Q4_FY_ResultsQA_Transcript.pdf" ,
                "Barclays_2025_Q1_ResultsQA_Transcript.pdf" ,
                "Barclays_2025_Q2_ResultsQA_Transcript.pdf"]
    asyncio.run(preproc_bank_documents(pdf_path,file_list))