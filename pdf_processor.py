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
from datetime import datetime
from documentation_model import Document, Page, Corpus, Fact, Question, upload_doc_to_neo4j

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


async def preproc_bank_documents(folder_path,file_list=None,focus=None,fact_label="FACT"):  
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
                await document.generate_document_1stfacts(focus=focus)

                # for page in document.pages:
                #     await page.embed_page_facts(model="text-embedding-3-large")
                await document.embed_document_facts(model="text-embedding-3-large")
                upload_doc_to_neo4j(corpus.neo4j_id,document,fact_label=fact_label)
        else:
            print(f"------------- File already exists: {file} ---------------------")


    corpus.upload_to_neo4j()


            #upload_to_neo4j(document)




if __name__ == "__main__":
    # pdf_path=r'C:\Users\volti\OneDrive\Documents\Python_projects\Dream_Team-Employeer_project\data\Barclays'
    # # for fpath in os.listdir(pdf_path):
    # #     print(fpath)
    # file_list=["Barclays_2022_Q1_ResultsQA_Transcript.pdf"
    #             "Barclays_2022_Q2_H1_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2022_Q3_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2022_Q4_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2023_Q1_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2023_Q2_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2023_Q3_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2024_Q1_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2024_Q2_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2024_Q3_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2024_Q4_FY_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2025_Q1_ResultsQA_Transcript.pdf" ,
    #             "Barclays_2025_Q2_ResultsQA_Transcript.pdf"]
    # asyncio.run(preproc_bank_documents(pdf_path,file_list))

    pdf_path=r'C:\Users\volti\OneDrive\Documents\Python_projects\Dream_Team-Employeer_project\data\PRA_Rulebooks'
    file_list=os.listdir(pdf_path)
    print(file_list)
    focus="You must focus on extracting facts that correspond to checks that the Bank of England must do in order to check if the bank is compliant with the PRA's rulebooks."
    asyncio.run(preproc_bank_documents(pdf_path,file_list,focus=focus,fact_label="RULE"))