# from fastapi import APIRouter, UploadFile, File
# from backend.core import hashing, chunking, milvus_ops, neo4j_ops

# router = APIRouter()

# @router.post("/")
# async def upload(file: UploadFile = File(...)):
#     try:
#         print(f"📤 Starting upload for: {file.filename}")
        
#         # Test Neo4j connection first
#         if not neo4j_ops.test_connection():
#             return {"status": "Failed", "error": "Neo4j connection failed"}
        
#         file_content = await file.read()
#         file_hash = hashing.compute_hash(file_content)
#         print(f"🔍 File hash: {file_hash}")

#         if hashing.hash_exists(file_hash):
#             print(f"⚠️ Duplicate file detected: {file.filename}")
#             return {"status": "Duplicate"}

#         print(f"📄 Chunking PDF: {file.filename}")
#         chunks = chunking.chunk_pdf(file_content)

#         if not chunks:
#             raise ValueError("No extractable text found in PDF.")

#         print(f"📊 Created {len(chunks)} chunks")
        
#         print(f"💾 Inserting chunks into Milvus...")
#         milvus_ops.insert_chunks(chunks, file.filename, file_hash)
        
#         print(f"🔗 Creating graph in Neo4j...")
#         neo4j_ops.create_knowledge_graph(chunks, file.filename)
        
#         print(f"💿 Saving hash to file...")
#         hashing.save_hash(file_hash, file.filename)

#         print(f"✅ Upload completed successfully for: {file.filename}")
#         return {"status": "Success"}

#     except ValueError as ve:
#         print(f"❌ Upload failed (ValueError): {ve}")
#         return {"status": "Failed", "error": str(ve)}

#     except Exception as e:
#         print(f"❌ Unexpected upload error: {e}")
#         return {"status": "Failed", "error": "Unexpected error occurred"}



# NEW CODE From Google

# backend/api/routes/upload.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.core import hashing, chunking, milvus_ops, neo4j_ops
import os # For potential domain determination

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)): # Renamed function for clarity
    try:
        print(f"📤 Starting upload for: {file.filename}")
        
        # Test Neo4j connection first - crucial before proceeding
        if not neo4j_ops.test_connection():
            print("❌ Neo4j connection failed at the start of upload.")
            # Return a proper HTTP error
            raise HTTPException(status_code=503, detail="Neo4j service unavailable. Cannot process upload.")
        
        file_content = await file.read()
        file_hash = hashing.compute_hash(file_content)
        print(f"🔍 File hash: {file_hash} for {file.filename}")

        # Check if document with this name or hash already processed to avoid re-processing (optional, depends on desired behavior)
        # This is a simple hash check; more robust would be to check Neo4j for document name
        # For example, you could check `neo4j_ops.get_document_stats()` for existing document.
        if hashing.hash_exists(file_hash):
            print(f"⚠️ Duplicate file content detected based on hash: {file.filename} (Hash: {file_hash})")
            # Consider what to return. Maybe an update confirmation or an error.
            return {"status": "Duplicate Content", "filename": file.filename, "message": "File content already processed."}

        print(f"📄 Chunking file: {file.filename}")
        # Assuming chunk_pdf handles various file types or you adapt it
        # If it's specific to PDF, you might want to check file.content_type
        if file.content_type == "application/pdf":
            chunks = chunking.chunk_pdf(file_content, filename=file.filename) # Pass filename for context if chunk_pdf uses it
        else:
            # Basic text splitting for other types, or add more sophisticated handlers
            # For simplicity, let's assume chunking.chunk_text exists for plain text
            try:
                text_content = file_content.decode('utf-8')
                chunks = chunking.chunk_text(text_content, chunk_size=1000, chunk_overlap=100) # Example
            except UnicodeDecodeError:
                 raise HTTPException(status_code=400, detail=f"Cannot decode file {file.filename} as text. Only PDF and plain text supported currently.")
            except AttributeError: # If chunking.chunk_text doesn't exist
                 raise HTTPException(status_code=501, detail=f"File type {file.content_type} not supported for chunking.")


        if not chunks:
            print(f"⚠️ No extractable text or chunks found in: {file.filename}")
            raise HTTPException(status_code=400, detail="No extractable text found in the uploaded file.")

        print(f"📊 Created {len(chunks)} chunks for {file.filename}")
        
        print(f"💾 Inserting {len(chunks)} chunks into Milvus for {file.filename}...")
        # Assuming insert_chunks handles embedding internally now
        milvus_ops.insert_chunks(chunks, file.filename, file_hash) # Pass file_hash if Milvus schema uses it
        
        print(f"🔗 Creating knowledge graph in Neo4j for {file.filename}...")
        # You can determine the domain based on filename, path, or user input if needed
        # For example, if files are in subdirectories like 'medical_papers/' or 'financial_reports/'
        file_domain = "general" # Default domain
        if "pharma" in file.filename.lower() or "medical" in file.filename.lower():
            file_domain = "biomedical"
        elif "finance" in file.filename.lower() or "report" in file.filename.lower():
            file_domain = "financial"
        # etc.
        
        neo4j_ops.create_knowledge_graph(chunks, file.filename, domain=file_domain)
        
        print(f"💿 Saving hash to file for {file.filename}...")
        hashing.save_hash_entry(file_hash, file.filename)

        print(f"✅ Upload completed successfully for: {file.filename}")
        return {
            "status": "Success",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "file_hash": file_hash,
            "domain_used_for_kg": file_domain
        }

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to be handled by FastAPI
        raise http_exc
    except ValueError as ve:
        print(f"❌ Upload failed (ValueError) for {file.filename}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"❌ Unexpected upload error for {file.filename}: {type(e).__name__} - {e}")
        # Log the full traceback here for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during upload: {type(e).__name__}")