import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import fitz
import re
# from docx import Document
import chromadb
import hashlib
import pickle

load_dotenv()

class Ingestion:
    def __init__(self, data_path, space_id, vector_store_name, config):
        self.data_path = data_path
        self.vector_store_name = vector_store_name
        self.config = config
        self.space_id = space_id
        # Initialize embedding model    
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        # Initialize ChromaDB client and collection
        dir_path = os.path.join("spaces", space_id)
        if not os.path.exists(dir_path):
            return {"error": "Space not found"}
    
        dir_path = os.path.join(dir_path, vector_store_name)
        self.dir_path = dir_path
        self.client = chromadb.PersistentClient(path=dir_path)
        self.collection = self.client.get_collection(self.vector_store_name)
        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size, 
                                                       chunk_overlap=self.config.overlap_size,
                                                       separators=["\n\n", "\n", " ", "?", "!", ","])

    def embed_text(self, text):
        return self.embeddings.embed_query(text)
    
    def merge_hypernated_words(self, text):
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(self, text):
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(self, text):
        return re.sub(r"\n{2,}", "\n", text)

    def clean_text(self, text):
        cleaning_functions = [
            self.merge_hypernated_words,
            self.fix_newlines,
            self.remove_multiple_newlines
        ]

        for cf in cleaning_functions:
            text = cf(text)

        return text

    def run_ingestion_pipeline(self):

        # chunking, metadata extraction and embedding
        all_chunks, all_embeddings, all_metadata, all_images = [], [], [], []
        # Get list of files from the data path
        if os.path.isdir(self.data_path):
            files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]
        else:
            files = [self.data_path]

        for file_path in files:

            # Generate a unique file version 
            with open(file_path, "rb") as f:
                file_bytes = f.read()
                file_version = hashlib.sha256(file_bytes).hexdigest()

            print(f"Processing file: {file_path} with version {file_version}")

            if file_path.lower().endswith(".pdf"):
                doc=fitz.open(file_path)

                for page_num, page in enumerate(doc):
                    ## process text
                    text=page.get_text()
                    if text.strip():
                        text = self.clean_text(text)
                        text_chunks = self.splitter.split_text(text)

                        #Embed each chunk using CLIP
                        for i, chunk in enumerate(text_chunks):
                            meta_data = {
                                "page_number": page_num,
                                "chunk_number": i+1,
                                "type": "text",
                                "file_version": file_version,
                                "file_name": os.path.basename(file_path),
                                "category": self.config.meta_data
                                }
                        
                            embedding = self.embeddings.embed_documents([chunk])
                            all_embeddings.append(embedding[0])
                            all_chunks.append(chunk)
                            all_metadata.append(meta_data)
                
                    ## process images
                    if self.config.multimodal:

                        image_list = page.get_images(full=True)
                        for img_index, img in enumerate(image_list):
                            try:
                                xref = img[0]
                                base_image = doc.extract_image(xref)
                                image_bytes = base_image["image"]
                                image_ext = base_image["ext"]
                                image_metadata = {
                                    "page": i,
                                    "type": "image",
                                    "file_version": file_version,
                                    "image_index": img_index,
                                    "image_ext": image_ext
                                }
                                all_images.append((image_bytes, image_metadata))    
                            except Exception as e:
                                print(f"Error extracting image {img_index} on page {i} of {file_path}: {e}")


            elif file_path.lower().endswith(".docx") or file_path.lower().endswith(".doc"):
                doc = Document(file_path)
                full_text = []  
                for para in doc.paragraphs:
                    full_text.append(para.text)
                    # If multimodal, extract images from docx
                    if self.config.multimodal:
                        for shape in para._element.xpath('.//w:drawing'):
                            # Try to extract images from drawing elements
                            blips = shape.xpath('.//a:blip', namespaces={'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'})
                            for blip in blips:
                                rId = blip.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                                if rId:
                                    image_part = doc.part.related_parts[rId]
                                    image_bytes = image_part.blob
                                    image_ext = os.path.splitext(image_part.partname)[1].lstrip('.')
                                    image_metadata = {
                                        "type": "image",
                                        "file_version": file_version,
                                        "image_ext": image_ext,
                                        "para_index": len(full_text) - 1
                                    }
                                    all_images.append((image_bytes, image_metadata))
                text = "\n".join(full_text)
                text = self.clean_text(text)
                text_doc = Document(page_content=text, metadata={"type": "text", "file_version": file_version})
                text_chunks = self.splitter.split_documents([text_doc])
                for chunk in text_chunks:
                    embedding = self.embed_text(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_chunks.append(chunk)

                
            elif file_path.lower().endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                text = self.clean_text(text)
                text_doc = Document(page_content=text, metadata={"type": "text", "file_version": file_version})
                text_chunks = self.splitter.split_documents([text_doc])
                for chunk in text_chunks:
                    embedding = self.embed_text(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_chunks.append(chunk)
 
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

            # Store embeddings in ChromaDB

            # Add embeddings to ChromaDB
            if all_embeddings:
                self.collection.add(
                    embeddings=all_embeddings,
                    documents=all_chunks,
                    metadatas=all_metadata,
                    ids=[f"{os.path.basename(file_path)}_{i}" for i in range(len(all_embeddings))]
                    )
                print(f"Added {len(all_embeddings)} embeddings from {file_path} to ChromaDB collection {self.vector_store_name}")

         # Get all raw documents from the ChromaDB
        raw_docs = self.collection.get(include=["documents", "metadatas"])

        # Convert them in Document object
        documents = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"])
        ]
        # Store the documents in dir_path as a pickle file

        documents_path = os.path.join(self.dir_path, "documents.pkl")
        with open(documents_path, "wb") as f:
            pickle.dump(documents, f)
        print(f"Stored {len(documents)} documents to {documents_path}")        
       