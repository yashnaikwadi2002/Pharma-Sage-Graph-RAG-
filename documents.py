# from fastapi import APIRouter
# import json
# import os

# router = APIRouter()

# @router.get("/documents/")
# def list_documents():
#     """
#     List all uploaded documents with their status
#     """
#     try:
#         # Read from hashes.json to get uploaded files
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return []
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Convert hash data to document list
#         documents = []
#         for file_hash, filename in hashes.items():
#             documents.append({
#                 "name": filename,
#                 "status": "processed",  # All files in hashes.json are processed
#                 "hash": file_hash
#             })
        
#         return documents
        
#     except Exception as e:
#         print(f"Error listing documents: {e}")
#         return [{"name": "Error loading documents", "status": "error"}]

# @router.get("/documents/{filename}")
# def get_document_info(filename: str):
#     """
#     Get information about a specific document
#     """
#     try:
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return {"error": "No documents found"}
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Find document by filename
#         for file_hash, stored_filename in hashes.items():
#             if stored_filename == filename:
#                 return {
#                     "name": filename,
#                     "status": "processed",
#                     "hash": file_hash
#                 }
        
#         return {"error": "Document not found"}
        
#     except Exception as e:
#         return {"error": f"Error retrieving document info: {str(e)}"}

# @router.delete("/documents/{filename}")
# def delete_document(filename: str):
#     """
#     Delete a specific document from all storage systems
#     """
#     try:
#         from backend.core import hashing, milvus_ops, neo4j_ops
#         import urllib.parse
        
#         # URL decode the filename in case it contains special characters
#         filename = urllib.parse.unquote(filename)
        
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return {"error": "No documents found"}
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Find and remove document
#         file_hash_to_remove = None
#         for file_hash, stored_filename in hashes.items():
#             if stored_filename == filename:
#                 file_hash_to_remove = file_hash
#                 break
        
#         if not file_hash_to_remove:
#             return {"error": f"Document '{filename}' not found"}
        
#         # Remove from Milvus
#         try:
#             milvus_ops.delete_by_hash(file_hash_to_remove)
#             print(f"Removed {filename} from Milvus")
#         except Exception as e:
#             print(f"Warning: Could not remove from Milvus: {e}")
        
#         # Remove from Neo4j
#         try:
#             neo4j_ops.delete_document_graph(filename)
#             print(f"Removed {filename} from Neo4j")
#         except Exception as e:
#             print(f"Warning: Could not remove from Neo4j: {e}")
        
#         # Remove from hashes
#         del hashes[file_hash_to_remove]
#         with open(hash_file, 'w') as f:
#             json.dump(hashes, f, indent=2)
        
#         return {
#             "message": f"Document '{filename}' deleted successfully",
#             "deleted_hash": file_hash_to_remove
#         }
        
#     except Exception as e:
#         print(f"Error deleting document {filename}: {e}")
#         return {"error": f"Error deleting document: {str(e)}"}

# @router.get("/documents/stats/")
# def get_document_stats():
#     """
#     Get statistics about uploaded documents
#     """
#     try:
#         from backend.core import neo4j_ops
        
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return {
#                 "total_documents": 0,
#                 "total_chunks": 0,
#                 "document_details": []
#             }
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         total_documents = len(hashes)
#         total_chunks = 0
#         document_details = []
        
#         # Get chunk counts from Neo4j
#         with neo4j_ops.driver.session() as session:
#             for file_hash, filename in hashes.items():
#                 result = session.run(
#                     "MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk) RETURN count(c) as chunk_count",
#                     filename=filename
#                 )
#                 record = result.single()
#                 chunk_count = record["chunk_count"] if record else 0
#                 total_chunks += chunk_count
                
#                 document_details.append({
#                     "name": filename,
#                     "hash": file_hash,
#                     "chunks": chunk_count
#                 })
        
#         return {
#             "total_documents": total_documents,
#             "total_chunks": total_chunks,
#             "document_details": document_details
#         }
        
#     except Exception as e:
#         return {"error": f"Error getting document stats: {str(e)}"}

# NEW CODE from CLAUDE>>><><><><><<><<><><<><<><<><<><><><><><><><><><><><><><><><><<><><><><><><>

# from fastapi import APIRouter, HTTPException
# import json
# import os
# from typing import List, Dict, Optional

# router = APIRouter()

# @router.get("/documents/")
# def list_documents():
#     """
#     List all uploaded documents with their status and enhanced metadata
#     """
#     try:
#         # Import here to avoid circular imports
#         from backend.core import neo4j_ops
        
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return []
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Get enhanced stats from Neo4j
#         doc_stats = neo4j_ops.get_document_stats()
        
#         documents = []
#         for file_hash, filename in hashes.items():
#             # Find matching document stats
#             doc_info = next((d for d in doc_stats if d["document"] == filename), None)
            
#             documents.append({
#                 "name": filename,
#                 "status": "processed",
#                 "hash": file_hash,
#                 "total_chunks": doc_info.get("total_chunks", 0) if doc_info else 0,
#                 "processed_chunks": doc_info.get("processed_chunks", 0) if doc_info else 0,
#                 "unique_entities": doc_info.get("unique_entities", 0) if doc_info else 0,
#                 "entity_mentions": doc_info.get("entity_mentions", 0) if doc_info else 0,
#                 "relationships": doc_info.get("relationships", 0) if doc_info else 0,
#                 "created": doc_info.get("created") if doc_info else None,
#                 "completed": doc_info.get("completed") if doc_info else None
#             })
        
#         return documents
        
#     except Exception as e:
#         print(f"Error listing documents: {e}")
#         return [{"name": "Error loading documents", "status": "error"}]

# @router.get("/documents/{filename}")
# def get_document_info(filename: str):
#     """
#     Get detailed information about a specific document including entities and relationships
#     """
#     try:
#         from backend.core import neo4j_ops
#         import urllib.parse
        
#         # URL decode the filename
#         filename = urllib.parse.unquote(filename)
        
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return {"error": "No documents found"}
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Find document by filename
#         file_hash = None
#         for stored_hash, stored_filename in hashes.items():
#             if stored_filename == filename:
#                 file_hash = stored_hash
#                 break
        
#         if not file_hash:
#             return {"error": "Document not found"}
        
#         # Get detailed stats from Neo4j
#         with neo4j_ops.driver.session() as session:
#             # Get basic document info
#             doc_query = """
#             MATCH (d:Document {name: $filename})
#             OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
#             OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
#             RETURN d.name as document_name,
#                    d.chunk_count as total_chunks,
#                    d.entity_count as entity_mentions,
#                    d.relationship_count as relationships,
#                    count(DISTINCT c) as processed_chunks,
#                    count(DISTINCT e) as unique_entities,
#                    d.created_at as created,
#                    d.processing_completed_at as completed
#             """
            
#             result = session.run(doc_query, filename=filename).single()
            
#             if not result:
#                 return {"error": "Document not found in knowledge graph"}
            
#             # Get entity types distribution
#             entity_types_query = """
#             MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
#             RETURN e.type as entity_type, count(DISTINCT e) as count
#             ORDER BY count DESC
#             """
            
#             entity_types = session.run(entity_types_query, filename=filename)
#             entity_distribution = [dict(record) for record in entity_types]
            
#             # Get most mentioned entities
#             top_entities_query = """
#             MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
#             WITH e, count(c) as mention_count
#             RETURN e.name as entity_name, e.type as entity_type, 
#                    e.description as description, mention_count
#             ORDER BY mention_count DESC
#             LIMIT 10
#             """
            
#             top_entities = session.run(top_entities_query, filename=filename)
#             top_entities_list = [dict(record) for record in top_entities]
            
#             # Get relationship types for this document
#             rel_types_query = """
#             MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e1:Entity)
#             MATCH (e1)-[r]-(e2:Entity)
#             WHERE any(doc IN r.source_documents WHERE doc = $filename)
#             RETURN type(r) as relationship_type, count(DISTINCT r) as count
#             ORDER BY count DESC
#             """
            
#             rel_types = session.run(rel_types_query, filename=filename)
#             relationship_distribution = [dict(record) for record in rel_types]
        
#         return {
#             "name": filename,
#             "status": "processed",
#             "hash": file_hash,
#             "statistics": {
#                 "total_chunks": result["total_chunks"],
#                 "processed_chunks": result["processed_chunks"],
#                 "unique_entities": result["unique_entities"],
#                 "entity_mentions": result["entity_mentions"],
#                 "relationships": result["relationships"],
#                 "created": str(result["created"]) if result["created"] else None,
#                 "completed": str(result["completed"]) if result["completed"] else None
#             },
#             "entity_distribution": entity_distribution,
#             "top_entities": top_entities_list,
#             "relationship_distribution": relationship_distribution
#         }
        
#     except Exception as e:
#         print(f"Error getting document info: {e}")
#         return {"error": f"Error retrieving document info: {str(e)}"}

# @router.delete("/documents/{filename}")
# def delete_document(filename: str):
#     """
#     Delete a specific document from all storage systems with enhanced cleanup
#     """
#     try:
#         from backend.core import milvus_ops, neo4j_ops
#         import urllib.parse
        
#         # URL decode the filename
#         filename = urllib.parse.unquote(filename)
        
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return {"error": "No documents found"}
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Find document hash
#         file_hash_to_remove = None
#         for file_hash, stored_filename in hashes.items():
#             if stored_filename == filename:
#                 file_hash_to_remove = file_hash
#                 break
        
#         if not file_hash_to_remove:
#             return {"error": f"Document '{filename}' not found"}
        
#         deletion_log = []
        
#         # Remove from Milvus
#         try:
#             milvus_ops.delete_by_hash(file_hash_to_remove)
#             deletion_log.append("✅ Removed from Milvus vector store")
#             print(f"Removed {filename} from Milvus")
#         except Exception as e:
#             deletion_log.append(f"⚠️ Warning: Could not remove from Milvus: {e}")
#             print(f"Warning: Could not remove from Milvus: {e}")
        
#         # Remove from Neo4j (enhanced deletion with cleanup)
#         try:
#             neo4j_ops.delete_document_graph(filename)
#             deletion_log.append("✅ Removed from Neo4j knowledge graph with orphan cleanup")
#             print(f"Removed {filename} from Neo4j")
#         except Exception as e:
#             deletion_log.append(f"⚠️ Warning: Could not remove from Neo4j: {e}")
#             print(f"Warning: Could not remove from Neo4j: {e}")
        
#         # Remove from hashes
#         del hashes[file_hash_to_remove]
#         with open(hash_file, 'w') as f:
#             json.dump(hashes, f, indent=2)
        
#         deletion_log.append("✅ Removed from document registry")
        
#         return {
#             "message": f"Document '{filename}' deleted successfully",
#             "deleted_hash": file_hash_to_remove,
#             "deletion_log": deletion_log
#         }
        
#     except Exception as e:
#         print(f"Error deleting document {filename}: {e}")
#         return {"error": f"Error deleting document: {str(e)}"}

# @router.get("/documents/stats/")
# def get_document_stats():
#     """
#     Get comprehensive statistics about uploaded documents and knowledge graph
#     """
#     try:
#         from backend.core import neo4j_ops
        
#         hash_file = "hashes.json"
        
#         if not os.path.exists(hash_file):
#             return {
#                 "total_documents": 0,
#                 "total_chunks": 0,
#                 "total_entities": 0,
#                 "total_relationships": 0,
#                 "document_details": [],
#                 "entity_types": [],
#                 "relationship_types": [],
#                 "top_entities": []
#             }
        
#         with open(hash_file, 'r') as f:
#             hashes = json.load(f)
        
#         # Get comprehensive stats from Neo4j
#         graph_stats = neo4j_ops.get_graph_statistics()
#         doc_stats = neo4j_ops.get_document_stats()
        
#         basic_stats = graph_stats.get("basic_stats", {})
        
#         return {
#             "total_documents": len(hashes),
#             "total_chunks": basic_stats.get("chunks", 0),
#             "total_entities": basic_stats.get("entities", 0),
#             "entity_types": graph_stats.get("entity_types", []),
#             "relationship_types": graph_stats.get("relationship_types", []),
#             "top_entities": graph_stats.get("top_entities", []),
#             "document_details": doc_stats
#         }
        
#     except Exception as e:
#         print(f"Error getting document stats: {e}")
#         return {"error": f"Error getting document stats: {str(e)}"}

# @router.get("/documents/{filename}/entities")
# def get_document_entities(filename: str, entity_type: Optional[str] = None, limit: int = 50):
#     """
#     Get entities mentioned in a specific document, optionally filtered by type
#     """
#     try:
#         from backend.core import neo4j_ops
#         import urllib.parse
        
#         filename = urllib.parse.unquote(filename)
        
#         with neo4j_ops.driver.session() as session:
#             if entity_type:
#                 query = """
#                 MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
#                 WHERE e.type = $entity_type
#                 WITH e, count(c) as mention_count
#                 RETURN e.name as name, e.type as type, e.description as description, 
#                        mention_count, e.source_documents as source_documents
#                 ORDER BY mention_count DESC
#                 LIMIT $limit
#                 """
#                 result = session.run(query, filename=filename, entity_type=entity_type.upper(), limit=limit)
#             else:
#                 query = """
#                 MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
#                 WITH e, count(c) as mention_count
#                 RETURN e.name as name, e.type as type, e.description as description, 
#                        mention_count, e.source_documents as source_documents
#                 ORDER BY mention_count DESC
#                 LIMIT $limit
#                 """
#                 result = session.run(query, filename=filename, limit=limit)
            
#             entities = [dict(record) for record in result]
            
#             return {
#                 "document": filename,
#                 "entity_type_filter": entity_type,
#                 "entities": entities,
#                 "total_found": len(entities)
#             }
            
#     except Exception as e:
#         return {"error": f"Error getting entities for document: {str(e)}"}

# @router.get("/documents/{filename}/relationships")
# def get_document_relationships(filename: str, limit: int = 50):
#     """
#     Get relationships found in a specific document
#     """
#     try:
#         from backend.core import neo4j_ops
#         import urllib.parse
        
#         filename = urllib.parse.unquote(filename)
        
#         with neo4j_ops.driver.session() as session:
#             query = """
#             MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e1:Entity)
#             MATCH (e1)-[r]-(e2:Entity)
#             WHERE any(doc IN r.source_documents WHERE doc = $filename)
#             RETURN e1.name as source_entity, type(r) as relationship_type, 
#                    e2.name as target_entity, r.description as description,
#                    e1.type as source_type, e2.type as target_type,
#                    r.weight as strength
#             ORDER BY r.weight DESC
#             LIMIT $limit
#             """
            
#             result = session.run(query, filename=filename, limit=limit)
#             relationships = [dict(record) for record in result]
            
#             return {
#                 "document": filename,
#                 "relationships": relationships,
#                 "total_found": len(relationships)
#             }
            
#     except Exception as e:
#         return {"error": f"Error getting relationships for document: {str(e)}"}

# @router.get("/knowledge-graph/search")
# def search_knowledge_graph(query: str, limit: int = 10):
#     """
#     Search the knowledge graph for entities and relationships
#     """
#     try:
#         from backend.core import neo4j_ops
        
#         if not query or len(query.strip()) < 2:
#             return {"error": "Query must be at least 2 characters long"}
        
#         results = neo4j_ops.query_knowledge_graph(query.strip(), limit)
        
#         return {
#             "query": query,
#             "results": results,
#             "timestamp": "now"  # You can add actual timestamp if needed
#         }
        
#     except Exception as e:
#         return {"error": f"Error searching knowledge graph: {str(e)}"}

# @router.get("/knowledge-graph/entity/{entity_name}")
# def get_entity_details(entity_name: str):
#     """
#     Get detailed information about a specific entity including all its relationships
#     """
#     try:
#         from backend.core import neo4j_ops
#         import urllib.parse
        
#         entity_name = urllib.parse.unquote(entity_name)
        
#         with neo4j_ops.driver.session() as session:
#             # Get entity details
#             entity_query = """
#             MATCH (e:Entity {name: $entity_name})
#             RETURN e.name as name, e.type as type, e.description as description,
#                    e.mention_count as mentions, e.source_documents as source_documents
#             """
            
#             entity_result = session.run(entity_query, entity_name=entity_name).single()
            
#             if not entity_result:
#                 return {"error": f"Entity '{entity_name}' not found"}
            
#             # Get all relationships
#             relationships_query = """
#             MATCH (e:Entity {name: $entity_name})-[r]-(related:Entity)
#             RETURN related.name as related_entity, related.type as related_type,
#                    type(r) as relationship_type, r.description as relationship_description,
#                    r.weight as strength, r.source_documents as relationship_sources
#             ORDER BY r.weight DESC
#             LIMIT 20
#             """
            
#             rel_result = session.run(relationships_query, entity_name=entity_name)
#             relationships = [dict(record) for record in rel_result]
            
#             # Get document contexts where this entity appears
#             context_query = """
#             MATCH (c:Chunk)-[:MENTIONS]->(e:Entity {name: $entity_name})
#             MATCH (d:Document)-[:CONTAINS]->(c)
#             RETURN d.name as document, c.chunk_index as chunk_index,
#                    c.content[0..200] as context_snippet
#             ORDER BY d.name, c.chunk_index
#             LIMIT 10
#             """
            
#             context_result = session.run(context_query, entity_name=entity_name)
#             contexts = [dict(record) for record in context_result]
        
#         return {
#             "entity": dict(entity_result),
#             "relationships": relationships,
#             "contexts": contexts,
#             "relationship_count": len(relationships),
#             "context_count": len(contexts)
#         }
        
#     except Exception as e:
#         return {"error": f"Error getting entity details: {str(e)}"}

# @router.delete("/documents/clear_all/")
# def clear_all_documents():
#     """
#     Completely delete all documents from Milvus, Neo4j, and the local registry.
#     WARNING: This action is irreversible!
#     """
#     try:
#         from backend.core import milvus_ops, neo4j_ops
#         import os

#         deletion_log = []
        
#         # Clear Neo4j graph
#         try:
#             with neo4j_ops.driver.session() as session:
#                 result = session.run("MATCH (n) RETURN count(n) as node_count").single()
#                 node_count = result["node_count"] if result else 0
                
#                 session.run("MATCH (n) DETACH DELETE n")
#                 deletion_log.append(f"✅ Cleared Neo4j knowledge graph ({node_count} nodes)")
#         except Exception as e:
#             deletion_log.append(f"⚠️ Warning: Could not clear Neo4j: {e}")
        
#         # Drop and recreate Milvus collection
#         try:
#             milvus_ops.drop_collection()
#             deletion_log.append("✅ Dropped Milvus collection")
#         except Exception as e:
#             deletion_log.append(f"⚠️ Warning: Could not drop Milvus collection: {e}")
        
#         # Delete hashes.json
#         try:
#             if os.path.exists("hashes.json"):
#                 with open("hashes.json", 'r') as f:
#                     hashes = json.load(f)
#                 doc_count = len(hashes)
#                 os.remove("hashes.json")
#                 deletion_log.append(f"✅ Removed document registry ({doc_count} documents)")
#             else:
#                 deletion_log.append("ℹ️ No document registry found")
#         except Exception as e:
#             deletion_log.append(f"⚠️ Warning: Could not remove document registry: {e}")
        
#         return {
#             "message": "All documents and metadata cleared successfully",
#             "deletion_log": deletion_log
#         }
    
#     except Exception as e:
#         return {"error": f"Error clearing all data: {str(e)}"}

# @router.get("/knowledge-graph/stats")
# def get_knowledge_graph_stats():
#     """
#     Get detailed statistics about the knowledge graph
#     """
#     try:
#         from backend.core import neo4j_ops
        
#         stats = neo4j_ops.get_graph_statistics()
#         return stats
        
#     except Exception as e:
#         return {"error": f"Error getting knowledge graph stats: {str(e)}"}


## NEW CODE GOOGLE <><><><><><><><><><><><><><<><><><<><><><><><><<><><<><><><><><><><><><><<><><><><>

# backend/api/routes/documents.py

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Optional, Any
from backend.core import neo4j_ops # Assuming this is the KG-enabled version
from backend.core import milvus_ops # For deletion if needed
import urllib.parse
import traceback
import os # For hashes.json if still used for deletion key mapping
import json # For hashes.json
from pydantic import BaseModel

router = APIRouter()

# --- Pydantic Models (Optional but good for response consistency) ---
class DocumentSummary(BaseModel):
    name: str
    status: Optional[str] = "unknown"
    # hash_value: Optional[str] = None # If you maintain hashes.json and want to show it
    total_chunks_in_doc: Optional[int] = None
    chunks_processed_for_kg: Optional[int] = None
    unique_entities_mentioned: Optional[int] = None
    created_at: Optional[str] = None
    kg_processing_completed_at: Optional[str] = None
    error_message: Optional[str] = None

class DocumentDetail(DocumentSummary):
    entity_type_distribution: Optional[List[Dict[str, Any]]] = None
    top_mentioned_entities: Optional[List[Dict[str, Any]]] = None
    relationship_type_distribution: Optional[List[Dict[str, Any]]] = None

class DeletionResponse(BaseModel):
    message: str
    details: Dict[str, Any]


# --- Document Endpoints ---
# @router.get("/", response_model=List[DocumentSummary], summary="List All Processed Documents")
# async def list_all_documents_endpoint():
#     """
#     Lists all documents that have been processed and ingested into the Neo4j knowledge graph.
#     Relies on `neo4j_ops.get_document_stats()` as the source of truth.
#     """
#     try:
#         doc_stats_from_neo4j = neo4j_ops.get_document_stats() # This is synchronous
        
#         if not doc_stats_from_neo4j:
#             return []
            
#         results: List[DocumentSummary] = []
#         for doc_stat in doc_stats_from_neo4j:
#             results.append(DocumentSummary(
#                 name=doc_stat.get("document"),
#                 status=doc_stat.get("status", "processed"),
#                 total_chunks_in_doc=doc_stat.get("total_chunks_in_doc") or doc_stat.get("total_chunks"),
#                 chunks_processed_for_kg=doc_stat.get("chunks_processed_for_kg") or doc_stat.get("processed_chunks"),
#                 unique_entities_mentioned=doc_stat.get("unique_entities_mentioned") or doc_stat.get("unique_entities"),
#                 created_at=str(doc_stat.get("created_at")) if doc_stat.get("created_at") else None,
#                 kg_processing_completed_at=str(doc_stat.get("kg_processing_completed_at") or doc_stat.get("entity_extraction_completed_at")) if (doc_stat.get("completed_at") or doc_stat.get("entity_extraction_completed_at")) else None,
#                 error_message=doc_stat.get("error")
#             ))
#         return results
        
#     except Exception as e:
#         print(f"❌ Error listing documents: {type(e).__name__} - {e}")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail="Failed to list documents from the knowledge graph.")

@router.get("/", response_model=List[DocumentSummary], summary="List All Processed Documents")
async def list_all_documents_endpoint():
    """
    Lists all documents that have been processed and ingested into the Neo4j knowledge graph.
    Relies on `neo4j_ops.get_document_stats()` as the source of truth.
    """
    try:
        # neo4j_ops.get_document_stats() is synchronous based on its current implementation
        doc_stats_from_neo4j = neo4j_ops.get_document_stats() 
        
        if not doc_stats_from_neo4j:
            return [] # Return empty list if no stats are found
            
        results: List[DocumentSummary] = []
        for doc_stat in doc_stats_from_neo4j:
            # Ensure all .get() calls use the exact aliases returned by get_document_stats()
            results.append(DocumentSummary(
                name=doc_stat.get("document"), # From: d.name AS document
                status=doc_stat.get("status"), # From: d.status AS status
                total_chunks_in_doc=doc_stat.get("total_chunks_in_doc"), # From: d.chunk_count AS total_chunks_in_doc
                chunks_actually_in_graph=doc_stat.get("chunks_actually_in_graph"), # From: count(DISTINCT c) AS chunks_actually_in_graph
                chunks_processed_for_kg=doc_stat.get("chunks_processed_for_kg"), # From: d.chunks_processed_for_entities AS chunks_processed_for_kg
                unique_entities_mentioned=doc_stat.get("unique_entities_mentioned"), # From: count(DISTINCT e) AS unique_entities_mentioned
                created_at=str(doc_stat.get("created_at")) if doc_stat.get("created_at") else None, # From: d.created_at AS created_at
                kg_processing_completed_at=str(doc_stat.get("kg_processing_completed_at")) if doc_stat.get("kg_processing_completed_at") else None, # From: d.entity_extraction_completed_at AS kg_processing_completed_at
                error_message=doc_stat.get("error_message") # From: d.error_message AS error_message
            ))
        return results
        
    except Exception as e:
        print(f"❌ Error listing documents: {type(e).__name__} - {e}")
        traceback.print_exc() # Log the full traceback for server-side debugging
        raise HTTPException(status_code=500, detail="Failed to list documents from the knowledge graph.")

@router.get("/{filename}/details", response_model=DocumentDetail, summary="Get Detailed Info for a Document")
async def get_single_document_info_endpoint(filename: str):
    """
    Retrieves detailed information for a specific document, including its statistics,
    entity distribution, top mentioned entities, and relationship types from the knowledge graph.
    """
    try:
        decoded_filename = urllib.parse.unquote(filename)
        
        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            doc_info_res = session.run("""
                MATCH (d:Document {name: $filename})
                RETURN d.name AS name, 
                       d.chunk_count AS total_chunks_in_doc, // Use this one
                       d.status AS status,
                       d.created_at AS created_at, 
                       d.entity_extraction_completed_at AS kg_processing_completed_at,
                       d.error_message AS error_message,
                       d.total_entities_processed AS total_entities_processed, // from KG processing
                       d.total_relationships_processed AS total_relationships_processed, // from KG processing
                       d.chunks_processed_for_entities AS chunks_processed_for_kg // from KG processing
            """, filename=decoded_filename).single()

            if not doc_info_res:
                raise HTTPException(status_code=404, detail=f"Document '{decoded_filename}' not found in the knowledge graph.")

            doc_data = dict(doc_info_res) # Convert to dict for easier access

            # Get entity types distribution for this document
            entity_types = session.run("""
                MATCH (d:Document {name: $filename})-[:CONTAINS]->(:Chunk)-[:MENTIONS]->(e)
                UNWIND labels(e) as entity_label
                WHERE NOT entity_label IN ['Entity', 'Chunk', 'Document'] // Exclude generic/structural labels
                RETURN entity_label AS type, count(DISTINCT e) AS count
                ORDER BY count DESC LIMIT 10
            """, filename=decoded_filename).data() # .data() converts all records to list of dicts
            
            top_entities = session.run("""
                MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e)
                WITH e, count(DISTINCT c) AS mention_count_in_doc // Count distinct chunks mentioning entity
                RETURN e.name AS name, labels(e) AS types, e.description AS description, mention_count_in_doc
                ORDER BY mention_count_in_doc DESC LIMIT 10
            """, filename=decoded_filename).data()
            
            rel_types = session.run("""
                MATCH (d:Document {name: $filename})-[:CONTAINS]->(:Chunk)-[:MENTIONS]->(e1)
                MATCH (e1)-[r]-(e2)
                // Ensure the relationship itself is relevant to this document if r.source_documents exists
                WHERE (NOT EXISTS(r.source_documents) OR $filename IN r.source_documents) 
                      AND NOT type(r) IN ['CONTAINS', 'NEXT', 'MENTIONS'] // Exclude structural relationships
                RETURN type(r) AS type, count(DISTINCT r) AS count
                ORDER BY count DESC LIMIT 10
            """, filename=decoded_filename).data()
        
        return DocumentDetail(
            name=doc_data.get("name"),
            status=doc_data.get("status"),
            total_chunks_in_doc=doc_data.get("total_chunks_in_doc"),
            chunks_processed_for_kg=doc_data.get("chunks_processed_for_kg"),
            # unique_entities_mentioned needs a distinct count of entities for this doc
            # We can estimate or add another query if needed; for now, relying on doc_data
            unique_entities_mentioned=doc_data.get("total_entities_processed"), # Approximation
            created_at=str(doc_data.get("created_at")) if doc_data.get("created_at") else None,
            kg_processing_completed_at=str(doc_data.get("kg_processing_completed_at")) if doc_data.get("kg_processing_completed_at") else None,
            error_message=doc_data.get("error_message"),
            entity_type_distribution=entity_types,
            top_mentioned_entities=top_entities,
            relationship_type_distribution=rel_types
        )
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Error getting document info for '{filename}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving document info: {str(e)}")


@router.delete("/{filename}", response_model=DeletionResponse, summary="Delete a Specific Document")
async def delete_single_document_endpoint(filename: str):
    """
    Deletes a specific document and its associated data from Neo4j (KG parts)
    and Milvus (vector embeddings). Also attempts to remove from local `hashes.json` registry.
    """
    try:
        decoded_filename = urllib.parse.unquote(filename)
        
        hash_file_path = "hashes.json" # Path to your hash registry
        file_hash_to_remove = None
        hashes_data = {}
        log_details = {}

        if os.path.exists(hash_file_path):
            try:
                with open(hash_file_path, 'r') as f:
                    hashes_data = json.load(f)
                for h, stored_fn in hashes_data.items():
                    if stored_fn == decoded_filename:
                        file_hash_to_remove = h
                        break
            except Exception as e_hash_read:
                log_details["hash_registry_error"] = f"Could not read hashes.json: {str(e_hash_read)}"
        
        # Delete from Neo4j (this is the primary deletion point for graph data)
        try:
            neo4j_delete_summary = neo4j_ops.delete_document_graph(decoded_filename) # Synchronous
            log_details["neo4j_deletion"] = neo4j_delete_summary or "Deletion attempted."
        except Exception as e_neo:
            log_details["neo4j_deletion_error"] = str(e_neo)
            print(f"❌ Error deleting '{decoded_filename}' from Neo4j: {e_neo}")
            # Decide if you want to proceed if Neo4j deletion fails. Maybe not.
            # raise HTTPException(status_code=500, detail=f"Failed to delete document from Neo4j: {str(e_neo)}")


        # Delete from Milvus (if hash was found and Milvus deletion is by hash)
        if file_hash_to_remove:
            try:
                if hasattr(milvus_ops, 'delete_by_hash') and callable(milvus_ops.delete_by_hash):
                    milvus_ops.delete_by_hash(file_hash_to_remove) # Assumes this function exists
                    log_details["milvus_deletion"] = f"Deletion attempted for hash '{file_hash_to_remove}'."
                    print(f"ℹ️ Milvus deletion attempted for {decoded_filename} (hash: {file_hash_to_remove})")
                else:
                    log_details["milvus_deletion"] = "Skipped: milvus_ops.delete_by_hash not available."
            except Exception as e_milvus:
                log_details["milvus_deletion_error"] = str(e_milvus)
                print(f"⚠️ Warning: Could not remove from Milvus for {decoded_filename}: {e_milvus}")
        elif not file_hash_to_remove:
            log_details["milvus_deletion"] = "Skipped: Document hash not found in registry for Milvus deletion."
        

        # Remove from hashes.json registry
        if file_hash_to_remove and os.path.exists(hash_file_path) and file_hash_to_remove in hashes_data:
            try:
                del hashes_data[file_hash_to_remove]
                with open(hash_file_path, 'w') as f:
                    json.dump(hashes_data, f, indent=2)
                log_details["hash_registry_update"] = f"Removed '{decoded_filename}' from local hash registry."
            except Exception as e_hash_write:
                log_details["hash_registry_error"] = f"Could not update hashes.json: {str(e_hash_write)}"
        elif os.path.exists(hash_file_path) and not file_hash_to_remove:
             log_details["hash_registry_update"] = f"Document '{decoded_filename}' not found in hash registry, no update needed."


        return DeletionResponse(
            message=f"Deletion process initiated for document '{decoded_filename}'. Review details.",
            details=log_details
        )
        
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Error deleting document '{filename}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Core error during document deletion: {str(e)}")


@router.get("/{filename}/entities", summary="Get Entities Mentioned in a Document")
async def get_document_entities_endpoint(
    filename: str,
    entity_type: Optional[str] = Query(None, description="Filter by a specific entity type (e.g., 'PERSON', 'DRUG'). Case-insensitive."),
    limit: int = Query(25, ge=1, le=100)
):
    """
    Retrieves entities mentioned within a specific document.
    Can be filtered by entity type. Entity types are typically the labels applied in Neo4j (e.g., PERSON, DRUG).
    """
    try:
        decoded_filename = urllib.parse.unquote(filename)
        
        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            params = {"filename": decoded_filename, "limit": limit}
            
            # Base query to find entities mentioned in the document
            cypher_query_parts = [
                "MATCH (d:Document {name: $filename})-[:CONTAINS]->(:Chunk)-[:MENTIONS]->(e)"
            ]
            
            # Add type filter if provided
            # The new neo4j_ops creates specific labels like :PERSON, :ORGANIZATION
            # So, we match against that specific label.
            safe_entity_type = ""
            if entity_type:
                # Sanitize to prevent Cypher injection, though labels are less risky than direct string concat in WHERE
                # A better way might be to have a known list of valid entity types.
                safe_entity_type = re.sub(r'[^a-zA-Z0-9_]', '', entity_type.strip().upper())
                if not safe_entity_type: # Or if not in a predefined list
                    raise HTTPException(status_code=400, detail="Invalid entity type format provided.")
                # Add label to the (e) node pattern
                cypher_query_parts[-1] = cypher_query_parts[-1].replace("(e)", f"(e:{safe_entity_type})")

            # Complete the query
            cypher_query_parts.extend([
                "WITH e, d", # Keep document context if needed for other aggregations
                "MATCH (e)<-[:MENTIONS]-(c:Chunk)<-[:CONTAINS]-(doc:Document) WHERE doc.name = $filename", # Ensure mentions are for THIS doc
                "WITH e, count(DISTINCT c) as mention_frequency_in_doc", # Count distinct chunks in THIS doc mentioning entity
                "RETURN e.name AS name,",
                "       labels(e) AS types,", # Get all labels
                "       e.description AS description,",
                "       mention_frequency_in_doc",
                "ORDER BY mention_frequency_in_doc DESC",
                "LIMIT $limit"
            ])
            
            final_cypher_query = "\n".join(cypher_query_parts)
            # print(f"Executing Cypher for entities: {final_cypher_query} with params: {params}") # For debugging
            
            result = session.run(final_cypher_query, **params)
            entities = [dict(record) for record in result]
            
        return {
            "document": decoded_filename,
            "entity_type_filter": safe_entity_type if entity_type else "ALL",
            "entities_returned": entities,
            "count_returned": len(entities)
        }
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Error getting entities for document '{filename}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting entities for document: {str(e)}")


@router.get("/{filename}/relationships", summary="Get Relationships Involving Entities from a Document")
async def get_document_relationships_endpoint(filename: str, limit: int = Query(25, ge=1, le=100)):
    """
    Retrieves relationships where at least one of the connected entities
    is mentioned within the specified document.
    """
    try:
        decoded_filename = urllib.parse.unquote(filename)

        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            # This query finds relationships connected to entities mentioned in the specified document.
            # It ensures that the relationship itself is sourced from this document if such info exists.
            cypher_query = """
            MATCH (doc:Document {name: $filename})-[:CONTAINS]->(:Chunk)-[:MENTIONS]->(e1) // e1 is mentioned in the doc
            MATCH (e1)-[r]-(e2)
            WHERE (NOT EXISTS(r.source_documents) OR $filename IN r.source_documents) // Relationship is relevant to this doc
              AND NOT type(r) IN ['CONTAINS', 'NEXT', 'MENTIONS'] // Exclude structural/meta relationships
            WITH e1, r, e2, doc // Pass document for context
            RETURN DISTINCT
                   e1.name AS entity1_name, labels(e1) AS entity1_types,
                   type(r) AS relationship_type,
                   e2.name AS entity2_name, labels(e2) AS entity2_types,
                   r.description AS relationship_description,
                   r.weight AS strength,
                   r.source_documents AS relationship_all_source_docs
            ORDER BY strength DESC NULLS LAST, entity1_name, entity2_name // Handle null strength
            LIMIT $limit
            """
            result = session.run(cypher_query, filename=decoded_filename, limit=limit)
            relationships = [dict(record) for record in result]
            
        return {
            "document": decoded_filename,
            "relationships_returned": relationships,
            "count_returned": len(relationships)
        }
            
    except Exception as e:
        print(f"❌ Error getting relationships for document '{filename}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error getting relationships for document: {str(e)}")


# Endpoint for clearing all data - this is the DANGEROUS one.
# It was in your "NEW CODE from Claude".
# Moved to /admin/ path and tagged appropriately.
@router.delete("/admin/clear-all-data",
    tags=["Admin - Dangerous Operations"],
    summary="DANGEROUS: Clear All Documents and Graph Data",
    response_model=DeletionResponse
)
async def admin_clear_all_data_endpoint():
    """
    WARNING: This action is irreversible!
    Completely deletes all documents from Milvus, all data from Neo4j,
    and attempts to clear the local `hashes.json` document registry.
    Should be protected by authentication/authorization in a real application.
    """
    print("‼️ WARNING: Initiating complete data wipe of Milvus, Neo4j, and hash registry.")
    log_details = {}
    
    # Add proper authentication here for such a destructive endpoint.
    # Example: if not current_user.is_admin: raise HTTPException(403, "Not authorized")

    # Clear Neo4j graph
    try:
        if neo4j_ops.driver:
            with neo4j_ops.driver.session(database="neo4j") as session:
                node_count_res = session.run("MATCH (n) RETURN count(n) as count").single()
                nodes_before = node_count_res["count"] if node_count_res else 0
                session.run("MATCH (n) DETACH DELETE n") # Deletes all nodes and relationships
                log_details["neo4j_deletion"] = f"Neo4j graph cleared. Nodes deleted: {nodes_before}."
        else:
            log_details["neo4j_deletion"] = "Skipped: Neo4j driver not initialized."
    except Exception as e_neo:
        log_details["neo4j_deletion_error"] = str(e_neo)
        print(f"❌ Error clearing Neo4j graph: {e_neo}")

    # Drop Milvus collection
    try:
        if hasattr(milvus_ops, 'drop_collection') and callable(milvus_ops.drop_collection):
            milvus_ops.drop_collection() # Assumes this function exists and handles collection presence
            log_details["milvus_action"] = "Milvus collection dropped (if it existed)."
            # You might need to re-initialize the collection after this via an init script or another endpoint
            # if hasattr(milvus_ops, 'create_collection_if_not_exists'): milvus_ops.create_collection_if_not_exists()
        else:
            log_details["milvus_action"] = "Skipped: milvus_ops.drop_collection not available."
    except Exception as e_milvus:
        log_details["milvus_action_error"] = str(e_milvus)
        print(f"❌ Error dropping Milvus collection: {e_milvus}")
    
    # Delete hashes.json registry
    hash_file_path = "hashes.json"
    if os.path.exists(hash_file_path):
        try:
            os.remove(hash_file_path)
            log_details["hash_registry_deletion"] = f"Local hash registry '{hash_file_path}' deleted."
        except Exception as e_hash:
            log_details["hash_registry_deletion_error"] = str(e_hash)
            print(f"❌ Error deleting hash registry '{hash_file_path}': {e_hash}")
    else:
        log_details["hash_registry_deletion"] = f"Local hash registry '{hash_file_path}' not found, no action taken."
        
    return DeletionResponse(
        message="DANGEROUS OPERATION COMPLETED: Attempted to clear all data.",
        details=log_details
    )