# from dotenv import load_dotenv
# load_dotenv()

# from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import numpy as np
# import os

# # Milvus (Zilliz Cloud) connection
# MILVUS_URI = os.getenv("MILVUS_URI")
# MILVUS_USER = os.getenv("MILVUS_USER")
# MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")

# connections.connect(
#     alias="default",
#     uri=MILVUS_URI,
#     user=MILVUS_USER,
#     password=MILVUS_PASSWORD,
#     secure=True
# )

# COLLECTION_NAME = "pharma_docs"

# # 1. Create collection (if not exists)
# def create_collection():
#     if COLLECTION_NAME in utility.list_collections():
#         return Collection(COLLECTION_NAME)

#     fields = [
#         FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64),
#         FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128, is_primary=True),
#         FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # Gemini embedding size
#     ]
#     schema = CollectionSchema(fields)
#     collection = Collection(name=COLLECTION_NAME, schema=schema)

#     collection.create_index(
#         field_name="embedding",
#         index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
#     )
#     collection.load()
#     return collection

# # 2. Insert multiple chunks
# def insert_chunks(chunks, filename, file_hash):
#     collection = create_collection()

#     # Embed the chunks using Gemini
#     embedder = GoogleGenerativeAIEmbeddings(
#         model="models/embedding-001",
#         google_api_key=os.getenv("GEMINI_API_KEY")
#     )
#     embeddings = embedder.embed_documents(chunks)

#     chunk_ids = [f"{file_hash}_{i}" for i in range(len(chunks))]

#     data = [
#         [file_hash] * len(chunks),
#         chunk_ids,
#         embeddings
#     ]

#     collection.insert(data)
#     collection.flush()

# # 3. Insert one embedding (optional)
# def insert_embedding(file_hash, vector):
#     collection = create_collection()
#     chunk_id = f"{file_hash}_single"
#     collection.insert([[file_hash], [chunk_id], [vector]])

# # 4. Search similar vectors
# def search_similar(vector, top_k=1):
#     collection = Collection(COLLECTION_NAME)
#     collection.load()
#     results = collection.search(
#         data=[vector],
#         anns_field="embedding",
#         param={"metric_type": "L2", "params": {"nprobe": 10}},
#         limit=top_k,
#         output_fields=["file_hash", "chunk_id"]
#     )
#     return results

# # 5. Delete all chunks of a file
# def delete_by_hash(file_hash):
#     collection = Collection(COLLECTION_NAME)
#     expr = f"file_hash == '{file_hash}'"
#     collection.delete(expr)

# def delete_hash_entry(file_hash):
#     delete_by_hash(file_hash)

# # 6. Drop entire collection (⚠️ DANGER)
# def drop_collection():
#     if COLLECTION_NAME in Collection.list_collections():
#         Collection(COLLECTION_NAME).drop()
#         print("✅ Dropped existing Milvus collection.")
#     else:
#         print("ℹ️ Collection not found.")

## NEW CODE GOOGLE <><<><><><><><><>><><><><><><><><><><><><><><><><<><><><><><><><><><><><><><><><<<><>><><><

# backend/core/milvus_ops.py

from dotenv import load_dotenv
load_dotenv() # Ensures .env is loaded when this module is imported

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import time
from typing import List, Optional, Dict # Ensure all used types are imported

# --- Milvus (Zilliz Cloud) Connection ---
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "pharma_docs_kg")
MILVUS_DIMENSION = 768 # Gemini "models/embedding-001" embedding size

# --- Global Variables ---
_milvus_collection_instance = None # Renamed to avoid conflict with Collection type
_embedder_instance = None # Renamed
_milvus_connected_flag = False # Flag to track if connection was attempted in this process

def _get_embedder() -> GoogleGenerativeAIEmbeddings:
    """Initializes and returns the embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            # This should ideally prevent app from starting or be caught by health check
            raise ValueError("CRITICAL: GEMINI_API_KEY not found in environment variables for Milvus embedder.")
        _embedder_instance = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
    return _embedder_instance

def connect_to_milvus(retry_count=3, delay=5) -> bool:
    """Establishes connection to Milvus with retries. Sets a flag on success."""
    global _milvus_connected_flag
    
    # If already successfully connected in this process, don't try again unless forced
    if _milvus_connected_flag and connections.has_connection("default"):
        # print("DEBUG: Milvus connection 'default' already active.")
        return True

    if not all([MILVUS_URI, MILVUS_USER, MILVUS_PASSWORD]):
        print("❌ ERROR: Milvus connection details (URI, USER, PASSWORD) incomplete in environment variables.")
        return False 
    
    print(f"ℹ️ Attempting to establish Milvus connection (URI: {MILVUS_URI}, User: {MILVUS_USER})...")
    for attempt in range(retry_count):
        try:
            connections.connect(
                alias="default",
                uri=MILVUS_URI,
                user=MILVUS_USER,       # Assumes MILVUS_USER is the username
                password=MILVUS_PASSWORD, # Assumes MILVUS_PASSWORD is the password/token
                                          # For Zilliz API Key: user="token", password="YOUR_API_KEY" OR token="YOUR_API_KEY"
                secure=True,
                timeout=20 # Added connection timeout
            )
            print(f"✅ Successfully connected to Milvus on attempt {attempt + 1}.")
            _milvus_connected_flag = True
            return True
        except Exception as e:
            print(f"⚠️ Milvus connection attempt {attempt + 1}/{retry_count} failed: {type(e).__name__} - {e}")
            if attempt < retry_count - 1:
                print(f"Retrying Milvus connection in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ All Milvus connection attempts failed.")
                _milvus_connected_flag = False # Explicitly set to false
                return False
    return False # Should not be reached if logic is correct

def get_or_create_collection() -> Collection:
    """
    Gets the existing collection or creates it if not present.
    Ensures Milvus connection is active before proceeding.
    Handles collection loading and index creation.
    """
    global _milvus_collection_instance
    
    if not _milvus_connected_flag or not connections.has_connection("default"):
        print("ℹ️ Milvus connection not active, attempting to connect for get_or_create_collection...")
        if not connect_to_milvus(): # Attempt to connect
            raise ConnectionError("CRITICAL: Failed to establish Milvus connection. Cannot proceed with collection operations.")

    # Check if the cached instance is valid and points to an existing server-side collection
    if _milvus_collection_instance is not None and _milvus_collection_instance.name == COLLECTION_NAME:
        if utility.has_collection(COLLECTION_NAME, using="default"):
            if not _is_collection_loaded(_milvus_collection_instance):
                print(f"ℹ️ Collection '{COLLECTION_NAME}' exists but not loaded. Loading...")
                _milvus_collection_instance.load()
            _create_index_if_not_exists(_milvus_collection_instance) # Also check index
            return _milvus_collection_instance
        else:
            print(f"ℹ️ Cached collection '{COLLECTION_NAME}' no longer exists on server. Re-initializing.")
            _milvus_collection_instance = None # Force re-initialization

    # If no valid cached instance, fetch or create
    if utility.has_collection(COLLECTION_NAME, using="default"):
        print(f"ℹ️ Collection '{COLLECTION_NAME}' found on server. Accessing existing.")
        _milvus_collection_instance = Collection(COLLECTION_NAME, using="default")
    else:
        print(f"ℹ️ Collection '{COLLECTION_NAME}' not found. Creating new collection.")
        fields = [
            FieldSchema(name="file_hash", dtype=DataType.VARCHAR, max_length=64, description="SHA256 hash of the original file content"),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=255, is_primary=True, auto_id=False, description="Unique ID for the chunk (e.g., file_hash_chunk_index)"),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255, description="Original filename of the document"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=MILVUS_DIMENSION, description="Vector embedding of the chunk content")
        ]
        schema = CollectionSchema(fields, description="Collection for pharma document chunks.", enable_dynamic_field=False)
        _milvus_collection_instance = Collection(name=COLLECTION_NAME, schema=schema, using="default", consistency_level="Strong")
        print(f"✅ Collection '{COLLECTION_NAME}' schema created.")
    
    _create_index_if_not_exists(_milvus_collection_instance)
    if not _is_collection_loaded(_milvus_collection_instance):
        print(f"ℹ️ Loading collection '{COLLECTION_NAME}' into memory.")
        _milvus_collection_instance.load()
    
    return _milvus_collection_instance

def _is_collection_loaded(collection: Collection) -> bool:
    """Checks if the collection is loaded by checking query segment info."""
    try:
        # A more direct way if available: utility.get_loading_progress might give info
        # For now, checking segment states.
        query_seg_info = utility.get_query_segment_info(collection.name, using="default")
        # If there's any segment info and at least one segment is loaded, assume loaded.
        # This can be refined based on specific Milvus version behavior.
        return len(query_seg_info) > 0 # Simplified check; loaded segments usually mean collection is loaded.
    except Exception as e:
        print(f"⚠️ Could not determine load status for collection '{collection.name}': {e}")
        return False # Assume not loaded if status check fails

def _create_index_if_not_exists(collection: Collection):
    """Creates an index on the embedding field if it doesn't exist."""
    if not collection.has_index(index_name="embedding_idx"): # Check by specific index name if you define one
        print(f"ℹ️ Index 'embedding_idx' not found for field 'embedding' in '{collection.name}'. Creating...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT", 
            "params": {"nlist": 128} 
        }
        try:
            collection.create_index(
                field_name="embedding", 
                index_params=index_params,
                index_name="embedding_idx" # Good practice to name your index
            )
            # Milvus often requires waiting for index to build
            utility.wait_for_index_building_complete(collection.name, index_name="embedding_idx", using="default")
            print(f"✅ Index 'embedding_idx' created and built for 'embedding' field.")
        except Exception as e:
            print(f"❌ Failed to create index for '{collection.name}': {e}")
            # Decide if this should be a critical failure
    # else:
        # print(f"ℹ️ Index 'embedding_idx' for 'embedding' field already exists in '{collection.name}'.")


def insert_chunks(chunks: List[str], filename: str, file_hash: str):
    """Embeds and inserts document chunks into Milvus."""
    if not chunks:
        print("⚠️ No chunks provided to insert_chunks. Skipping Milvus insertion.")
        return {"status": "skipped", "message": "No chunks to insert."}

    try:
        collection = get_or_create_collection() # This ensures connection and collection readiness
        embedder = _get_embedder()
    except ConnectionError as ce: # Catch connection error from get_or_create_collection
        print(f"❌ Milvus connection error during insert_chunks setup: {ce}")
        raise RuntimeError(f"Milvus connection failed, cannot insert chunks for {filename}: {ce}")
    except ValueError as ve: # Catch API key error from _get_embedder
        print(f"❌ Embedder initialization error: {ve}")
        raise RuntimeError(f"Embedder setup failed, cannot insert chunks for {filename}: {ve}")
    except Exception as e_setup:
        print(f"❌ Unexpected setup error in insert_chunks for '{filename}': {e_setup}")
        raise RuntimeError(f"Milvus/Embedder setup failed for {filename}: {e_setup}")


    print(f"ℹ️ Generating embeddings for {len(chunks)} chunks from '{filename}'...")
    try:
        embeddings = embedder.embed_documents(chunks)
    except Exception as e_embed:
        print(f"❌ Failed to generate embeddings for '{filename}': {e_embed}")
        raise RuntimeError(f"Embedding generation failed for '{filename}': {e_embed}")

    chunk_ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
    filenames_list = [filename] * len(chunks)

    data_to_insert = [
        [file_hash] * len(chunks),
        chunk_ids,
        filenames_list,
        embeddings
    ]

    print(f"ℹ️ Inserting {len(chunks)} chunk embeddings into Milvus collection '{COLLECTION_NAME}' for '{filename}'...")
    try:
        # Using upsert is generally safer if re-processing might occur.
        # It requires the primary key field (chunk_id) to be present.
        # If you are certain chunk_ids will always be new for new content, insert is fine.
        # Let's assume upsert for robustness if chunk_ids could repeat upon re-upload of modified file with same name.
        # insert_result = collection.insert(data_to_insert)
        upsert_result = collection.upsert(data_to_insert) # Upsert is generally safer for re-runs
        
        # Check upsert_result if it provides useful info (e.g., upsert_result.upsert_count)
        print(f"✅ Successfully upserted/inserted ~{len(chunk_ids)} entities into Milvus for '{filename}'. Result: {upsert_result.upsert_count if hasattr(upsert_result, 'upsert_count') else 'N/A'}.")
        collection.flush()
        print(f"ℹ️ Flushed collection '{COLLECTION_NAME}' after upsert/insertion for '{filename}'.")
        return {"status": "success", "inserted_count": len(chunk_ids)} # Or use upsert_result info
    except Exception as e_milvus_insert:
        print(f"❌ Milvus upsert/insertion error for '{filename}': {e_milvus_insert}")
        raise RuntimeError(f"Milvus upsert/insertion failed for '{filename}': {e_milvus_insert}")


def search_similar(query_vector: List[float], top_k: int = 5, filter_expr: Optional[str] = None) -> List[Dict]:
    """Searches for similar vectors in the collection. Returns a list of dicts."""
    try:
        collection = get_or_create_collection() # Ensures connection, collection, index, and load
    except ConnectionError as ce:
        print(f"❌ Milvus connection error during search_similar: {ce}")
        return [] # Return empty on connection failure
    except Exception as e_setup:
        print(f"❌ Unexpected setup error in search_similar: {e_setup}")
        return []

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 16}, 
    }

    # print(f"DEBUG: Milvus searching with top_k={top_k}, expr='{filter_expr if filter_expr else 'None'}'")
    try:
        raw_results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["file_hash", "chunk_id", "filename"],
            consistency_level="Strong" 
        )
        
        hits = []
        if raw_results and raw_results[0]:
            for hit_obj in raw_results[0]: # Iterate through Hit objects
                hit_data = {"id": hit_obj.id, "distance": hit_obj.distance}
                # Access fields from hit_obj.entity if they are simple types
                # For complex structures, to_dict() might be needed, but direct access is cleaner
                entity_fields = {}
                for field_name in ["file_hash", "chunk_id", "filename"]: # Fields we requested
                    if hasattr(hit_obj, field_name): # Pymilvus Hit object might have fields directly
                         entity_fields[field_name] = getattr(hit_obj, field_name)
                    elif hit_obj.entity and field_name in hit_obj.entity.fields:
                         entity_fields[field_name] = hit_obj.entity.get(field_name)

                hit_data["entity"] = entity_fields
                hits.append(hit_data)
        # print(f"DEBUG: Milvus search found {len(hits)} results.")
        return hits
    except Exception as e_search:
        print(f"❌ Milvus search error: {e_search}")
        return []


def delete_by_hash(file_hash: str) -> Dict:
    """Deletes all chunks associated with a specific file_hash."""
    if not file_hash:
        print("⚠️ delete_by_hash: file_hash cannot be empty.")
        return {"status": "error", "message": "file_hash cannot be empty."}

    try:
        collection = get_or_create_collection()
    except ConnectionError as ce:
        return {"status": "error", "message": f"Milvus connection failed: {ce}"}
    except Exception as e_setup:
        return {"status": "error", "message": f"Milvus setup failed for deletion: {e_setup}"}

    expr = f"file_hash == \"{file_hash}\"" # Ensure quotes for string comparison if file_hash can have spaces/special chars
    print(f"ℹ️ Attempting to delete entities from Milvus with expression: {expr}")
    try:
        # Optional: Query how many to delete for logging
        # query_res = collection.query(expr=expr, output_fields=["chunk_id"], limit=10000) # Max limit for query
        # num_to_delete = len(query_res)
        # print(f"ℹ️ Found {num_to_delete} entities in Milvus to delete for file_hash '{file_hash}'.")

        delete_result = collection.delete(expr)
        # delete_result typically contains MutationResult with delete_count
        deleted_count = delete_result.delete_count if hasattr(delete_result, 'delete_count') else 'N/A (check Milvus logs)'
        print(f"✅ Milvus deletion result for expr '{expr}': Count={deleted_count}")
        collection.flush()
        print(f"ℹ️ Flushed collection '{COLLECTION_NAME}' after deletion for hash '{file_hash}'.")
        return {"status": "success", "message": f"Deletion for file_hash '{file_hash}' processed. Matched/deleted count: {deleted_count}."}
    except Exception as e_delete:
        print(f"❌ Milvus deletion error for file_hash '{file_hash}': {e_delete}")
        return {"status": "error", "message": f"Milvus deletion failed for {file_hash}: {e_delete}"}


def drop_milvus_collection() -> Dict:
    """Drops the entire Milvus collection. USE WITH EXTREME CAUTION."""
    global _milvus_collection_instance, _milvus_connected_flag
    
    if not _milvus_connected_flag or not connections.has_connection("default"):
        print("ℹ️ Milvus connection not active, attempting to connect for drop_milvus_collection...")
        if not connect_to_milvus():
            return {"status": "error", "message": "Failed to connect to Milvus. Cannot drop collection."}

    if utility.has_collection(COLLECTION_NAME, using="default"):
        print(f"⚠️ DANGER: Attempting to drop Milvus collection '{COLLECTION_NAME}'...")
        try:
            collection_to_drop = Collection(COLLECTION_NAME, using="default") # Get handle to collection
            collection_to_drop.drop() # Use method on Collection object
            # utility.drop_collection(COLLECTION_NAME, using="default") # This also works
            _milvus_collection_instance = None # Reset global var
            print(f"✅ Successfully dropped Milvus collection '{COLLECTION_NAME}'.")
            return {"status": "success", "message": f"Collection '{COLLECTION_NAME}' dropped."}
        except Exception as e_drop:
            print(f"❌ Failed to drop Milvus collection '{COLLECTION_NAME}': {e_drop}")
            return {"status": "error", "message": f"Failed to drop collection: {e_drop}"}
    else:
        print(f"ℹ️ Milvus collection '{COLLECTION_NAME}' not found. Nothing to drop.")
        return {"status": "not_found", "message": f"Collection '{COLLECTION_NAME}' not found."}

# --- Helper for app startup/shutdown or health check ---
def check_milvus_connection_and_collection() -> bool:
    """Checks connection and ensures the collection is ready. Returns True if OK."""
    try:
        if not connect_to_milvus(): # Try to connect
            return False
        get_or_create_collection() # Try to get/create collection (this also loads it and ensures index)
        print("✅ Milvus connection and collection setup verified successfully.")
        return True
    except Exception as e:
        print(f"❌ Milvus setup verification failed during check: {type(e).__name__} - {e}")
        return False

# Call connect_to_milvus at module import time for applications that don't use startup events
# However, for FastAPI with workers, it's better to connect on demand or via startup event.
# The current design calls connect_to_milvus() within get_or_create_collection().

if __name__ == "__main__":
    print("--- Milvus Ops Direct Test Script ---")
    # This will attempt to connect and set up collection when check_milvus_connection_and_collection is called
    if check_milvus_connection_and_collection():
        print(f"Milvus is ready with collection: {COLLECTION_NAME}")
        print(f"Collection instance: {_milvus_collection_instance}")
        
        # Further tests can be added here if needed
        # e.g., sample insert, search, delete
        
        # Example: Test search with a dummy vector
        # dummy_vector = [0.1] * MILVUS_DIMENSION 
        # print("\nTesting search with dummy vector...")
        # search_hits = search_similar(dummy_vector, top_k=1)
        # if search_hits:
        #     print(f"Found {len(search_hits)} dummy hit(s): {search_hits[0]}")
        # else:
        #     print("No hits found for dummy vector (as expected for an empty collection or random vector).")
            
    else:
        print("Milvus setup failed. Check URI, User, Password/Token, and network connectivity.")