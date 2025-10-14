from fastapi import FastAPI
from pydantic import BaseModel
import os
import time
from uuid import uuid4
from typing import Optional
from fastapi import Query
import shutil
import json
import chromadb
from datetime import datetime
from fastapi import File, UploadFile, Form, BackgroundTasks
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from src.ingestion import Ingestion
import pickle


app = FastAPI()

class Space(BaseModel):
    name: str
    description: str

class SpaceConfig(BaseModel):
    embedding_model: str = "text-embedding-ada-002"  # default value
    retrieval_model: str = "gpt-4-mini"  # default value
    embedding_dimension: int = 1536  # default value
    chunk_size: int = 768   # default value
    overlap_size: int = 100   # default value
    meta_data: Optional[str] = ""
    multimodal: Optional[bool] = False    

@app.post("/create-space")
async def create_space(space: Space):
    space_id = str(uuid4())
    dir_path = os.path.join("spaces", space_id)
    os.makedirs(dir_path, exist_ok=True)
    # Store metadata in a JSON file
    metadata = {
        "name": space.name,
        "description": space.description,
        "directory": dir_path,
        "status": "created"
    }
    metadata_path = os.path.join(dir_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return {
        "status": "created",
        "space": space,
        "space_id": space_id,
        "directory": dir_path
    }

@app.get("/spaces/{space_id}")
async def retrieve_space(space_id: str):
    dir_path = os.path.join("spaces", space_id)
    if not os.path.exists(dir_path):
        return {"error": "Space not found"}
    # Optionally, you could load metadata about the space here
    metadata_path = os.path.join(dir_path, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    return {
        "message": "Space retrieved successfully",
        "metadata": metadata,
        "space_id": space_id,
    }

@app.delete("/spaces/{space_id}")
async def delete_space(space_id: str):
    dir_path = os.path.join("spaces", space_id)
    if not os.path.exists(dir_path):
        return {"error": "Space not found"}
    try:
        shutil.rmtree(dir_path)
        return {
            "message": "Space deleted successfully",
            "space_id": space_id
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/spaces/{space_id}/vector-store")
async def create_vector_store(space_id: str, vector_store_name: str = Query(..., description="Name of the vector store")):

    dir_path = os.path.join("spaces", space_id)
    if not os.path.exists(dir_path):
        return {"error": "Space not found"}
    try:

        dir_path = os.path.join(dir_path, vector_store_name)
    
        db = chromadb.PersistentClient(path=dir_path)
        chroma_collection = db.create_collection(
            name=vector_store_name,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": 200
                }
            },
            metadata={
                "description": f"collection {vector_store_name} for space {space_id}",
                "created": str(datetime.now())
            })            
        
    except Exception as e:
        return {"error": str(e)}
    
    
    print(chroma_collection.id)

    return {
        "status": "Created",
        "vector_store": {
            "id": chroma_collection.id,
            "name": chroma_collection.name,
            "metadata": chroma_collection.metadata
        }
    }

@app.get("/spaces/{space_id}/vector-store")
async def get_vector_store(space_id: str, vector_store_name: str = Query(..., description="Name of the vector store")):

    dir_path = os.path.join("spaces", space_id)
    if not os.path.exists(dir_path):
        return {"error": "Space not found"}
    
    try:
        dir_path = os.path.join(dir_path, vector_store_name)
        db = chromadb.PersistentClient(path=dir_path)
        chroma_collection = db.get_collection(
            name=vector_store_name)            
        
    except Exception as e:
        return {"error": str(e)}
    
    return {
        "status": "Success",
        "vector_store": {
            "id": chroma_collection.id,
            "name": chroma_collection.name,
            "metadata": chroma_collection.metadata
        }
    }

@app.delete("/spaces/{space_id}/vector-store")
async def delete_vector_store(space_id: str, vector_store_name: str = Query(..., description="Name of the vector store")):
    dir_path = os.path.join("spaces", space_id, vector_store_name)

    print(dir_path)
    if not os.path.exists(dir_path):
        return {"error": "Space not found"}
    
    try:
        db = chromadb.PersistentClient(path=dir_path)
        db.delete_collection(name=vector_store_name)
        db.clear_system_cache()
        db = chromadb.PersistentClient(path=None)

        shutil.rmtree(dir_path, ignore_errors=True)

    except Exception as e:
        return {"error": str(e)}
    
    return {
        "status": "Deleted"
    }


@app.post("/spaces/{space_id}/vector-store/{vector_store_name}/ingest")
async def ingest(
    space_id: str,
    vector_store_name: str,
    background_tasks: BackgroundTasks,
    config: str = Form(default=json.dumps(SpaceConfig().dict())),
    file: UploadFile = File(None)
):
    dir_path = os.path.join("data", space_id)
    os.makedirs(dir_path, exist_ok=True)

    filename = None
    if file is not None:
        file_path = os.path.join(dir_path, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        filename = file.filename

    config_obj = SpaceConfig(**json.loads(config))
    ingestion = Ingestion(dir_path, space_id, vector_store_name, config_obj)

    def run_and_cleanup():
        ingestion.run_ingestion_pipeline()
        
        # Retry logic for deleting the directory if it's in use
        for _ in range(5):
            try:
                shutil.rmtree(dir_path)
                break
            except Exception as e:
                print(f"Retrying deletion due to error: {e}")
                time.sleep(1)
        else:
            print(f"Failed to delete {dir_path} after several attempts.")

    background_tasks.add_task(run_and_cleanup)

    return {
        "space_id": space_id,
        "vector_store_name": vector_store_name,
        "filename": filename,
        "config": config_obj
    }

@app.post("/spaces/{space_id}/vector-store/{vector_store_name}/similarity-search")
async def similarity_search(
    space_id: str,
    vector_store_name: str,
    query: str = Form(...),
    top_k: int = Form(3),
    meta_filter: Optional[str] = Form(None)
):
    dir_path = os.path.join("spaces", space_id, vector_store_name)
    if not os.path.exists(dir_path):
        return {"error": "Vector store not found"}
    try:
        
        where = None
        if meta_filter not in (None, "", "null"):
            where = meta_filter

        # Create Chroma instance
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = Chroma(
            collection_name=vector_store_name,
            embedding_function=embeddings,
            persist_directory=dir_path,
        )

        # Create retriever for similarity search
        similarity_search_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
            'k': top_k,
            'include': ['distances', 'metadatas', 'documents'],
            'filter': {"category": where}
            }
        )

        # Get relevant documents and their scores
        results = similarity_search_retriever.invoke(query)
        
        return {
            "results": results
        }
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/spaces/{space_id}/vector-store/{vector_store_name}/hybrid-search")
async def hybrid_search(
    space_id: str,
    vector_store_name: str,
    query: str = Form(...),
    top_k: int = Form(3)
):
    dir_path = os.path.join("spaces", space_id, vector_store_name)
    if not os.path.exists(dir_path):
        return {"error": "Vector store not found"}
    try:
        
        # Get the Documents from pickle file
        pickle_path = os.path.join(dir_path, "documents.pkl")
        if not os.path.exists(pickle_path):
            return {"error": "Pickle file with documents not found"}
        with open(pickle_path, "rb") as f:
            documents = pickle.load(f)

       # Create BM25Retriever from the documents
        bm25_retriever = BM25Retriever.from_documents(documents=documents, k=top_k)

        # Create Chroma instance
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vector_store = Chroma(
            collection_name=vector_store_name,
            embedding_function=embeddings,
            persist_directory=dir_path,
        )

      # Create vector search retriever from ChromaDB instance
        similarity_search_retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={'k': top_k}
            )
       # Ensemble the retrievers using Langchain’s EnsembleRetriever Object
        ensemble_retriever = EnsembleRetriever(retrievers=[similarity_search_retriever, bm25_retriever], weights=[0.5, 0.5])
        # Retrieve k relevant documents for the query
        return ensemble_retriever.invoke(query)
        
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/spaces/{space_id}/vector-store/{vector_store_name}/rag-fusion")
async def rag_fusion(
    space_id: str,
    vector_store_name: str,
    query: list = Form(...),
    top_k: int = Form(3)
):
    dir_path = os.path.join("spaces", space_id, vector_store_name)
    if not os.path.exists(dir_path):
        return {"error": "Vector store not found"}