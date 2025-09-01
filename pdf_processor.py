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

#Loading environment variables
dotenv.load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode('utf-8')
   
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
        self.keywords:list[str] = None

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
            async def generate_page_embedding(page):
                if page.summary is not None:
                    input_text = page.summary
                else:
                    input_text = page.text
                embedding_response = await client.embeddings.create(input=input_text, model=model)
                page.embedding = embedding_response.data[0].embedding
                
            
            # Generate all embeddings concurrently
            await asyncio.gather(*[generate_page_embedding(page) for page in self.pages])
        finally:
            await client.close()
        page_embeddings = [page.embedding for page in self.pages]
        self.embedding = average_normalize_and_format_embedding(page_embeddings)


    async def get_document_summaries(self, openai_model: str = "gpt-5-nano") -> tuple[str, str]:
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
        async def process_page(page):
            page_image_base64 = _image_to_base64(page.image)
            page_summary = await _send_images_to_openai([page_image_base64], page_summary_prompt.format(document=document_summary), openai_model)
            page_summary_json = json.loads(page_summary)
            page.summary = page_summary_json["summary"]
            page.keywords = page_summary_json["keywords"]
            return page_summary
        
        # Use asyncio.gather for concurrent processing
        page_summaries = await asyncio.gather(*[process_page(page) for page in self.pages])
        
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
        
        async def process_page_representation(page_idx, page):
            try:
                print(f"Detecting best representation for page {page_idx}/{len(self.pages)}")
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
        
        # Process all pages concurrently
        await asyncio.gather(*[process_page_representation(page_idx, page) for page_idx, page in enumerate(self.pages)])
    
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

def upload_to_neo4j(pdf_document:Document) -> None:
    """
    Upload the PDF to Neo4j.
    """
    print(f"Uploading PDF to Neo4j: {pdf_document.path}")
    from DB_neo4j import graph_store, EntityNode,Relation
    
    node_to_insert=[]

    relation_to_insert=[]
    document_node=EntityNode(name=pdf_document.name,
                            label="_Document",
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
    document_node_id=document_node.id


    for page in pdf_document.pages:
        page_node=EntityNode(name=page.page_id,label="_Page",properties={"text":page.text,
        #"image":page.image,
        "embedding":page.embedding,
        "summary":page.summary,
        "keywords":page.keywords,
        "image_url":page.image_url,
        "best_representation":page.best_representation})
        node_to_insert.append(page_node)
        page_node_id=page_node.id
        
        relation_to_insert.append(Relation(label="CONTAINS",properties={"page_id":page.page_id},source_id=document_node_id,target_id=page_node_id))
    
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


async def main():
    pdf_path=r'downloads\2025-q2-earnings-results-presentation.pdf'
    document=Document(path=pdf_path)
    await document.setup_from_path(image_container_path=Path("images"))

    await document.get_document_summaries()
    await document.detect_best_representation()
    upload_to_neo4j(document)




if __name__ == "__main__":
    asyncio.run(main())