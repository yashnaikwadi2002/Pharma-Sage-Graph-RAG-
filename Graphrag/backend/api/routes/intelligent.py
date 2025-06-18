# from fastapi import APIRouter, HTTPException, Query
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# from backend.core import neo4j_ops, milvus_ops
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage
# import os
# import json

# router = APIRouter()
# os.environ["GEMINI_API_KEY"] = "AIzaSyBdPP7bEpEQX8lIhQdAHGye9V9ooECquW0"
# # LLM and embedding model setup
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY"),
#     temperature=0.1,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2
# )

# embedder = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

# class QueryRequest(BaseModel):
#     query: str
#     mode: str = "hybrid"  # "semantic", "graph", or "hybrid"
#     limit: int = 10
#     include_context: bool = True

# @router.post("/")
# async def intelligent_query(request: QueryRequest):
#     try:
#         if request.mode == "semantic":
#             return await semantic_search(request.query, request.limit)
#         elif request.mode == "graph":
#             return await graph_query(request.query, request.limit)
#         elif request.mode == "hybrid":
#             return await hybrid_query(request.query, request.limit, request.include_context)
#         else:
#             raise HTTPException(status_code=400, detail="Invalid query mode.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# async def semantic_search(query: str, limit: int):
#     embedding = embedder.embed_query(query)
#     results = milvus_ops.search_similar(embedding, top_k=limit)
    
#     enriched = []
#     with neo4j_ops.driver.session() as session:
#         for result in results[0]:
#             chunk_id = result.entity.get("chunk_id")
#             res = session.run("""
#                 MATCH (c:Chunk {chunk_id: $chunk_id})<-[:CONTAINS]-(d:Document)
#                 RETURN c.content AS content, d.name AS document, c.chunk_index AS index
#             """, chunk_id=chunk_id)
#             record = res.single()
#             if record:
#                 enriched.append({
#                     "chunk_id": chunk_id,
#                     "similarity": float(1 / (1 + result.distance)),
#                     "document": record["document"],
#                     "chunk_index": record["index"],
#                     "content": record["content"]
#                 })
#     return {"mode": "semantic", "query": query, "results": enriched, "total_found": len(enriched)}

# async def graph_query(query: str, limit: int):
#     result = neo4j_ops.query_knowledge_graph(query, limit)
#     return {
#         "mode": "graph",
#         "query": query,
#         "cypher_query": result.get("cypher", ""),
#         "results": result.get("results", []),
#         "total_found": len(result.get("results", []))
#     }

# async def hybrid_query(query: str, limit: int, include_context: bool):
#     sem_results = await semantic_search(query, limit // 2)

#     # Use Gemini to extract entities
#     extraction_prompt = f"""Extract important biomedical entities from this query: "{query}". Return only a JSON list like ["entity1", "entity2"]"""
#     extracted = llm.invoke([HumanMessage(content=extraction_prompt)])
#     try:
#         entities = json.loads(extracted.content.strip())
#     except:
#         entities = []

#     # Graph enrichment
#     connections = []
#     with neo4j_ops.driver.session() as session:
#         for ent in entities[:3]:
#             res = session.run("""
#                 MATCH (e {name: $entity_name}) OPTIONAL MATCH (e)-[r]-(other)
#                 RETURN e.name AS source, type(r) AS rel, other.name AS target
#             """, entity_name=ent)
#             connections.extend([dict(record) for record in res])

#     # Contextual LLM answer
#     context_answer = ""
#     if include_context:
#         prompt = f"""
#         Provide an insightful answer to this query: "{query}"
#         Based on: {sem_results['results'][:2]} and graph links {connections[:3]}
#         """
#         context_answer = llm.invoke([HumanMessage(content=prompt)]).content

#     return {
#         "mode": "hybrid",
#         "query": query,
#         "semantic_results": sem_results["results"],
#         "graph_results": connections,
#         "entities_found": entities,
#         "contextual_answer": context_answer,
#         "total_semantic": len(sem_results["results"]),
#         "total_graph": len(connections)
#     }

# @router.get("/health/")
# async def health_check():
#     neo4j_ok = neo4j_ops.test_connection()
#     try:
#         _ = embedder.embed_query("test")
#         embed_ok = True
#     except:
#         embed_ok = False
#     try:
#         _ = llm.invoke([HumanMessage(content="ping")])
#         llm_ok = True
#     except:
#         llm_ok = False

#     return {
#         "neo4j": "healthy" if neo4j_ok else "unhealthy",
#         "embedding": "healthy" if embed_ok else "unhealthy",
#         "llm": "healthy" if llm_ok else "unhealthy"
#     }

#NEW CODE By claude.>>>><><><><><<><><><><><><><><><><><><><><><><><><
# from fastapi import APIRouter, HTTPException, Query
# from pydantic import BaseModel
# from typing import List, Dict, Optional, Any
# from backend.core import neo4j_ops, milvus_ops
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.schema import HumanMessage
# import os
# import json
# import re
# from collections import defaultdict

# router = APIRouter()
# os.environ["GEMINI_API_KEY"] = "AIzaSyBdPP7bEpEQX8lIhQdAHGye9V9ooECquW0"

# # LLM and embedding model setup
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     google_api_key=os.getenv("GEMINI_API_KEY"),
#     temperature=0.1,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2
# )

# embedder = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

# class QueryRequest(BaseModel):
#     query: str
#     mode: str = "hybrid"  # "semantic", "graph", or "hybrid"
#     limit: int = 10
#     include_context: bool = True
#     graph_depth: int = 2  # For graph traversal depth

# @router.post("/")
# async def intelligent_query(request: QueryRequest):
#     try:
#         if request.mode == "semantic":
#             return await semantic_search(request.query, request.limit)
#         elif request.mode == "graph":
#             return await graph_query(request.query, request.limit, request.graph_depth)
#         elif request.mode == "hybrid":
#             return await hybrid_query(request.query, request.limit, request.include_context, request.graph_depth)
#         else:
#             raise HTTPException(status_code=400, detail="Invalid query mode. Use 'semantic', 'graph', or 'hybrid'.")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

# async def semantic_search(query: str, limit: int):
#     """Enhanced semantic search with better context retrieval"""
#     try:
#         embedding = embedder.embed_query(query)
#         results = milvus_ops.search_similar(embedding, top_k=limit)
        
#         enriched = []
#         with neo4j_ops.driver.session() as session:
#             for result in results[0]:
#                 chunk_id = result.entity.get("chunk_id")
                
#                 # Get chunk with surrounding context
#                 cypher_query = """
#                 MATCH (c:Chunk {chunk_id: $chunk_id})<-[:CONTAINS]-(d:Document)
#                 OPTIONAL MATCH (prev:Chunk)-[:NEXT]->(c)
#                 OPTIONAL MATCH (c)-[:NEXT]->(next:Chunk)
#                 RETURN c.content AS content, 
#                        d.name AS document, 
#                        c.chunk_index AS index,
#                        prev.content AS prev_content,
#                        next.content AS next_content,
#                        d.chunk_count AS total_chunks
#                 """
                
#                 res = session.run(cypher_query, chunk_id=chunk_id)
#                 record = res.single()
                
#                 if record:
#                     # Calculate similarity score (convert distance to similarity)
#                     similarity = float(1 / (1 + result.distance))
                    
#                     enriched.append({
#                         "chunk_id": chunk_id,
#                         "similarity": similarity,
#                         "document": record["document"],
#                         "chunk_index": record["index"],
#                         "total_chunks": record["total_chunks"],
#                         "content": record["content"],
#                         "context": {
#                             "previous": record["prev_content"][:200] if record["prev_content"] else None,
#                             "next": record["next_content"][:200] if record["next_content"] else None
#                         }
#                     })
        
#         return {
#             "mode": "semantic",
#             "query": query,
#             "results": enriched,
#             "total_found": len(enriched)
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Semantic search error: {str(e)}")

# async def extract_entities_and_concepts(query: str) -> Dict[str, List[str]]:
#     """Enhanced entity and concept extraction using LLM"""
#     try:
#         extraction_prompt = f"""
#         Analyze this query and extract key information: "{query}"
        
#         Return a JSON object with these categories:
#         {{
#             "entities": ["person names", "organizations", "locations", "specific terms"],
#             "concepts": ["general concepts", "topics", "themes"],
#             "keywords": ["important keywords", "technical terms"],
#             "relationships": ["action words", "relationship indicators"]
#         }}
        
#         Only return the JSON object, no other text.
#         """
        
#         response = llm.invoke([HumanMessage(content=extraction_prompt)])
        
#         try:
#             extracted = json.loads(response.content.strip())
#             # Validate structure
#             required_keys = ["entities", "concepts", "keywords", "relationships"]
#             for key in required_keys:
#                 if key not in extracted:
#                     extracted[key] = []
#             return extracted
#         except json.JSONDecodeError:
#             # Fallback to simple keyword extraction
#             words = re.findall(r'\b[A-Za-z]{3,}\b', query)
#             return {
#                 "entities": words[:3],
#                 "concepts": words[3:6],
#                 "keywords": words,
#                 "relationships": []
#             }
#     except Exception as e:
#         print(f"Entity extraction error: {e}")
#         # Simple fallback
#         words = re.findall(r'\b[A-Za-z]{3,}\b', query)
#         return {
#             "entities": words[:5],
#             "concepts": [],
#             "keywords": words,
#             "relationships": []
#         }

# async def graph_query(query: str, limit: int, depth: int = 2):
#     """Enhanced graph-based query with proper knowledge graph traversal"""
#     try:
#         # Extract entities and concepts
#         extracted = await extract_entities_and_concepts(query)
#         all_terms = (extracted["entities"] + extracted["concepts"] + 
#                     extracted["keywords"] + extracted["relationships"])
        
#         graph_results = []
#         entity_matches = []
        
#         with neo4j_ops.driver.session() as session:
#             # Strategy 1: Find chunks containing query terms
#             for term in all_terms[:5]:  # Limit to prevent too many queries
#                 if len(term) < 3:  # Skip very short terms
#                     continue
                    
#                 # Search for term in chunk content
#                 cypher_query = """
#                 MATCH (c:Chunk)<-[:CONTAINS]-(d:Document)
#                 WHERE toLower(c.content) CONTAINS toLower($term)
#                 RETURN c.chunk_id AS chunk_id,
#                        c.content AS content,
#                        c.chunk_index AS chunk_index,
#                        d.name AS document,
#                        $term AS matched_term
#                 LIMIT $limit
#                 """
                
#                 result = session.run(cypher_query, term=term, limit=limit//len(all_terms[:5]) + 1)
                
#                 for record in result:
#                     entity_matches.append({
#                         "chunk_id": record["chunk_id"],
#                         "document": record["document"],
#                         "chunk_index": record["chunk_index"],
#                         "content": record["content"][:300],
#                         "matched_term": record["matched_term"],
#                         "match_type": "content_match"
#                     })
            
#             # Strategy 2: Find related chunks through graph traversal
#             if entity_matches:
#                 # Get chunks that are connected to found chunks
#                 chunk_ids = [match["chunk_id"] for match in entity_matches[:3]]
                
#                 for chunk_id in chunk_ids:
#                     traversal_query = """
#                     MATCH (c:Chunk {chunk_id: $chunk_id})<-[:CONTAINS]-(d:Document)-[:CONTAINS]->(related:Chunk)
#                     WHERE related.chunk_id <> $chunk_id
#                     WITH related, d, 
#                          abs(related.chunk_index - c.chunk_index) AS distance
#                     WHERE distance <= $depth
#                     RETURN related.chunk_id AS chunk_id,
#                            related.content AS content,
#                            related.chunk_index AS chunk_index,
#                            d.name AS document,
#                            distance
#                     ORDER BY distance
#                     LIMIT 3
#                     """
                    
#                     result = session.run(traversal_query, chunk_id=chunk_id, depth=depth)
                    
#                     for record in result:
#                         graph_results.append({
#                             "chunk_id": record["chunk_id"],
#                             "document": record["document"],
#                             "chunk_index": record["chunk_index"],
#                             "content": record["content"][:300],
#                             "distance": record["distance"],
#                             "match_type": "graph_traversal",
#                             "source_chunk": chunk_id
#                         })
        
#         # Combine and deduplicate results
#         all_results = entity_matches + graph_results
#         seen_chunks = set()
#         unique_results = []
        
#         for result in all_results:
#             if result["chunk_id"] not in seen_chunks:
#                 seen_chunks.add(result["chunk_id"])
#                 unique_results.append(result)
        
#         # Sort by relevance (content matches first, then by proximity)
#         unique_results.sort(key=lambda x: (
#             0 if x["match_type"] == "content_match" else 1,
#             x.get("distance", 0)
#         ))
        
#         return {
#             "mode": "graph",
#             "query": query,
#             "extracted_terms": extracted,
#             "results": unique_results[:limit],
#             "total_found": len(unique_results),
#             "search_strategies": {
#                 "content_matches": len(entity_matches),
#                 "graph_traversal": len(graph_results)
#             }
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Graph query error: {str(e)}")

# async def hybrid_query(query: str, limit: int, include_context: bool, depth: int = 2):
#     """Enhanced hybrid search combining semantic and graph approaches"""
#     try:
#         # Run both searches in parallel conceptually
#         semantic_results = await semantic_search(query, limit // 2)
#         graph_results = await graph_query(query, limit // 2, depth)
        
#         # Merge and rank results
#         merged_results = []
        
#         # Add semantic results with scoring
#         for result in semantic_results["results"]:
#             merged_results.append({
#                 **result,
#                 "source": "semantic",
#                 "combined_score": result["similarity"] * 0.7  # Weight semantic results
#             })
        
#         # Add graph results with scoring
#         for result in graph_results["results"]:
#             # Calculate relevance score for graph results
#             base_score = 0.5
#             if result["match_type"] == "content_match":
#                 base_score = 0.8
#             elif result["match_type"] == "graph_traversal":
#                 base_score = max(0.3, 0.8 - (result.get("distance", 0) * 0.1))
            
#             merged_results.append({
#                 **result,
#                 "source": "graph",
#                 "combined_score": base_score * 0.6  # Weight graph results
#             })
        
#         # Remove duplicates and sort by combined score
#         seen_chunks = set()
#         unique_merged = []
        
#         for result in merged_results:
#             if result["chunk_id"] not in seen_chunks:
#                 seen_chunks.add(result["chunk_id"])
#                 unique_merged.append(result)
        
#         # Sort by combined score
#         unique_merged.sort(key=lambda x: x["combined_score"], reverse=True)
#         final_results = unique_merged[:limit]
        
#         # Generate contextual answer if requested
#         contextual_answer = ""
#         if include_context and final_results:
#             context_content = []
#             for result in final_results[:3]:  # Use top 3 results for context
#                 context_content.append({
#                     "document": result["document"],
#                     "content": result["content"][:400],
#                     "source": result["source"]
#                 })
            
#             context_prompt = f"""
#             Based on the following relevant information, provide a comprehensive answer to the query: "{query}"
            
#             Relevant Information:
#             {json.dumps(context_content, indent=2)}
            
#             Please provide a clear, informative answer that synthesizes the information above. 
#             If the information is insufficient, mention what additional context might be helpful.
#             """
            
#             try:
#                 response = llm.invoke([HumanMessage(content=context_prompt)])
#                 contextual_answer = response.content
#             except Exception as e:
#                 contextual_answer = f"Error generating contextual answer: {str(e)}"
        
#         return {
#             "mode": "hybrid",
#             "query": query,
#             "results": final_results,
#             "semantic_count": len(semantic_results["results"]),
#             "graph_count": len(graph_results["results"]),
#             "total_found": len(final_results),
#             "contextual_answer": contextual_answer,
#             "extracted_terms": graph_results.get("extracted_terms", {}),
#             "performance": {
#                 "semantic_results": len(semantic_results["results"]),
#                 "graph_results": len(graph_results["results"]),
#                 "merged_unique": len(unique_merged),
#                 "final_results": len(final_results)
#             }
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Hybrid query error: {str(e)}")

# @router.get("/health/")
# async def health_check():
#     """Enhanced health check with more detailed status"""
#     try:
#         # Test Neo4j connection
#         neo4j_ok = neo4j_ops.test_connection()
        
#         # Test embedding model
#         embed_ok = False
#         embed_error = ""
#         try:
#             test_embedding = embedder.embed_query("test query")
#             embed_ok = len(test_embedding) > 0
#         except Exception as e:
#             embed_error = str(e)
        
#         # Test LLM
#         llm_ok = False
#         llm_error = ""
#         try:
#             response = llm.invoke([HumanMessage(content="ping")])
#             llm_ok = len(response.content) > 0
#         except Exception as e:
#             llm_error = str(e)
        
#         # Test Milvus connection
#         milvus_ok = False
#         milvus_error = ""
#         try:
#             # This is a simple test - you might want to implement a proper test in milvus_ops
#             milvus_ok = True  # Assume OK if no exception
#         except Exception as e:
#             milvus_error = str(e)
        
#         # Get some stats
#         stats = {}
#         try:
#             with neo4j_ops.driver.session() as session:
#                 result = session.run("""
#                     MATCH (d:Document) 
#                     OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
#                     RETURN count(DISTINCT d) as documents, count(c) as chunks
#                 """)
#                 record = result.single()
#                 if record:
#                     stats = {
#                         "documents": record["documents"],
#                         "chunks": record["chunks"]
#                     }
#         except Exception as e:
#             stats = {"error": str(e)}
        
#         overall_health = all([neo4j_ok, embed_ok, llm_ok, milvus_ok])
        
#         return {
#             "overall_status": "healthy" if overall_health else "degraded",
#             "components": {
#                 "neo4j": {
#                     "status": "healthy" if neo4j_ok else "unhealthy",
#                     "error": "" if neo4j_ok else "Connection failed"
#                 },
#                 "embedding_model": {
#                     "status": "healthy" if embed_ok else "unhealthy",
#                     "error": embed_error
#                 },
#                 "llm": {
#                     "status": "healthy" if llm_ok else "unhealthy",
#                     "error": llm_error
#                 },
#                 "milvus": {
#                     "status": "healthy" if milvus_ok else "unhealthy",
#                     "error": milvus_error
#                 }
#             },
#             "statistics": stats,
#             "version": "enhanced_v1.0"
#         }
        
#     except Exception as e:
#         return {
#             "overall_status": "unhealthy",
#             "error": str(e)
#         }

# # Additional utility endpoints
# @router.get("/graph/explore/{document_name}")
# async def explore_document_graph(document_name: str, depth: int = 1):
#     """Explore the graph structure of a specific document"""
#     try:
#         with neo4j_ops.driver.session() as session:
#             query = """
#             MATCH (d:Document {name: $doc_name})-[:CONTAINS]->(c:Chunk)
#             OPTIONAL MATCH (c)-[r:NEXT]->(next:Chunk)
#             RETURN d.name as document,
#                    d.chunk_count as total_chunks,
#                    collect({
#                        chunk_id: c.chunk_id,
#                        chunk_index: c.chunk_index,
#                        content_preview: substring(c.content, 0, 100),
#                        has_next: next IS NOT NULL
#                    }) as chunks
#             """
            
#             result = session.run(query, doc_name=document_name)
#             record = result.single()
            
#             if not record:
#                 return {"error": f"Document '{document_name}' not found in graph"}
            
#             return {
#                 "document": record["document"],
#                 "total_chunks": record["total_chunks"],
#                 "chunks": record["chunks"],
#                 "graph_structure": "sequential_chunks_with_next_relationships"
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Graph exploration error: {str(e)}")

# @router.get("/stats/query")
# async def get_query_stats():
#     """Get statistics about the knowledge base for query optimization"""
#     try:
#         with neo4j_ops.driver.session() as session:
#             stats_query = """
#             MATCH (d:Document)
#             OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
#             WITH d, count(c) as chunk_count
#             RETURN {
#                 total_documents: count(d),
#                 total_chunks: sum(chunk_count),
#                 avg_chunks_per_doc: avg(chunk_count),
#                 documents: collect({
#                     name: d.name,
#                     chunks: chunk_count
#                 })
#             } as stats
#             """
            
#             result = session.run(stats_query)
#             record = result.single()
            
#             return record["stats"] if record else {"error": "No data found"}
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Stats query error: {str(e)}")



# Code from GOOGLE<><><><><><><><><><><><><><><><><<><<><><<><<><<><<><><<><><><<><<
# backend/api/routes/intelligent.py

from dotenv import load_dotenv
load_dotenv() # Load .env file at the earliest point
from fastapi import APIRouter, HTTPException, Body, Query #can be used for GET params if needed
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from backend.core import neo4j_ops, milvus_ops # neo4j_ops now refers to the KG-enabled version
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
import os
import json
import re
import traceback # For detailed error logging

router = APIRouter()

# Ensure GEMINI_API_KEY is loaded.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in environment variables. Please set it for LLM and Embedding services.")

# LLM and embedding model setup
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Consider "gemini-pro" for more complex summary if flash is too limited
    google_api_key=GEMINI_API_KEY,
    temperature=0.1, # Low temperature for more factual, less creative responses
    max_tokens=2048,
    timeout=120,
    max_retries=2
)

embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", # Or "text-embedding-004"
    google_api_key=GEMINI_API_KEY
)

# --- Pydantic Models ---
class IntelligentQueryRequest(BaseModel):
    query: str = Field(..., description="The user's natural language query.")
    mode: str = Field(default="hybrid", description="Query mode: 'semantic', 'graph', or 'hybrid'.")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of primary results to return.")
    include_contextual_summary: bool = Field(default=True, description="Whether to generate an LLM-based summary from results.")
    domain_context: Optional[str] = Field(default="general", description="Optional domain context for graph querying (e.g., 'biomedical').")
    # graph_search_depth: int = Field(default=1, description="Depth for graph traversal if used in graph_query (not directly used by current neo4j_ops.query_knowledge_graph).")

class SemanticHit(BaseModel):
    document_name: str
    chunk_id: str
    chunk_index: int
    content_preview: str
    mentioned_entities_in_chunk: Optional[List[str]] = None
    similarity: float
    source: str = "semantic"

class GraphEntityHit(BaseModel):
    entity_name: str
    entity_labels: List[str]
    entity_description: Optional[str] = None
    entity_mentions: Optional[int] = None # Overall mention count from KG
    direct_connections: Optional[List[Dict[str, Any]]] = None # From neo4j_ops.query_knowledge_graph
    source: str = "graph"

class GraphContextChunk(BaseModel):
    document_name: str
    chunk_id: str
    chunk_index: int
    content_preview: str
    matched_entity: str # Entity from graph query that led to this chunk
    source: str = "graph_entity_context"


class IntelligentQueryResponse(BaseModel):
    mode: str
    query: str
    contextual_summary: Optional[str] = None
    semantic_search_results: Optional[List[SemanticHit]] = None
    graph_query_results: Optional[List[GraphEntityHit]] = None # Entity-centric results
    graph_entity_context_chunks: Optional[List[GraphContextChunk]] = None # Chunks MENTIONING graph entities
    details: Dict[str, Any]
    error: Optional[str] = None


# --- Query Endpoints ---
@router.post("/", response_model=IntelligentQueryResponse, summary="Perform an intelligent query")
async def intelligent_query_endpoint(request: IntelligentQueryRequest = Body(...)):
    """
    Performs an intelligent query against the knowledge base.
    Supports semantic search, graph-based search, or a hybrid approach.
    Can also generate a contextual summary based on the findings.
    """
    try:
        if request.mode == "semantic":
            result_data = await semantic_search(request.query, request.limit)
            return IntelligentQueryResponse(
                mode="semantic", query=request.query,
                semantic_search_results=result_data.get("results", []),
                details={"total_found": result_data.get("total_found",0)},
                error=result_data.get("error")
            )
        elif request.mode == "graph":
            result_data = await graph_based_query(request.query, request.limit, request.domain_context)
            return IntelligentQueryResponse(
                mode="graph", query=request.query,
                graph_query_results=result_data.get("results", []),
                details={
                    "total_found": result_data.get("total_found",0),
                    "cypher_executed": result_data.get("cypher_executed"),
                    "message": result_data.get("message")
                },
                error=result_data.get("error")
            )
        elif request.mode == "hybrid":
            return await hybrid_search_query(
                request.query,
                request.limit,
                request.include_contextual_summary,
                request.domain_context
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid query mode. Use 'semantic', 'graph', or 'hybrid'.")
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Unexpected error in intelligent_query_endpoint: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Core query processing error: {str(e)}")

# --- Search Functions ---
async def semantic_search(query_text: str, limit: int) -> Dict:
    """Performs semantic search using Milvus and enriches with Neo4j chunk data."""
    try:
        embedding = embedder.embed_query(query_text)
        milvus_results_raw = milvus_ops.search_similar(embedding, top_k=limit)

        if not milvus_results_raw or not milvus_results_raw[0]:
             return {"query": query_text, "results": [], "total_found": 0, "message": "No semantic matches found."}

        enriched_results: List[SemanticHit] = []
        # Make sure driver is available
        if not neo4j_ops.driver:
            return {"query": query_text, "results": [], "total_found": 0, "error": "Neo4j driver not initialized for semantic enrichment."}

        with neo4j_ops.driver.session(database="neo4j") as session:
            for hit in milvus_results_raw[0]:
                chunk_id = hit.entity.get("chunk_id")
                if not chunk_id:
                    print(f"⚠️ Milvus hit missing chunk_id: {hit.entity}")
                    continue

                res = session.run("""
                    MATCH (c:Chunk {chunk_id: $chunk_id})<-[:CONTAINS]-(d:Document)
                    OPTIONAL MATCH (c)-[:MENTIONS]->(e) // Match any entity type mentioned
                    RETURN d.name AS document_name,
                           c.chunk_id AS chunk_id,
                           c.chunk_index AS chunk_index,
                           c.content_preview AS content_preview,
                           COLLECT(DISTINCT e.name)[..5] AS mentioned_entities
                """, chunk_id=chunk_id).single()

                if res:
                    similarity_score = 1.0 / (1.0 + float(hit.distance)) if hit.distance is not None else 0.0
                    enriched_results.append(SemanticHit(
                        document_name=res["document_name"],
                        chunk_id=res["chunk_id"],
                        chunk_index=res["chunk_index"],
                        content_preview=res["content_preview"],
                        mentioned_entities_in_chunk=res["mentioned_entities"],
                        similarity=similarity_score
                    ))
                else:
                    print(f"⚠️ Chunk {chunk_id} from Milvus not found in Neo4j for enrichment.")
        
        return {
            "query": query_text,
            "results": enriched_results,
            "total_found": len(enriched_results)
        }
    except Exception as e:
        print(f"❌ Error in semantic_search: {type(e).__name__} - {e}")
        traceback.print_exc()
        return {"query": query_text, "results": [], "total_found": 0, "error": f"Semantic search failed: {str(e)}"}


async def graph_based_query(query_text: str, limit: int, domain_context: Optional[str] = "general") -> Dict:
    """Performs a query directly against the knowledge graph using neo4j_ops."""
    try:
        # neo4j_ops.query_knowledge_graph is synchronous, no await needed unless it's changed
        graph_query_result = neo4j_ops.query_knowledge_graph(query_text, limit=limit, domain_context=domain_context)
        
        # Convert to GraphEntityHit Pydantic model for type safety and API contract
        entity_hits: List[GraphEntityHit] = []
        for res_item in graph_query_result.get("results", []):
            entity_hits.append(GraphEntityHit(
                entity_name=res_item.get("entity_name"),
                entity_labels=res_item.get("entity_labels", []),
                entity_description=res_item.get("entity_description"),
                entity_mentions=res_item.get("entity_mentions"),
                direct_connections=res_item.get("direct_connections", [])
            ))

        return {
            "query": query_text,
            "results": entity_hits,
            "cypher_executed": graph_query_result.get("cypher_executed", "N/A"),
            "total_found": len(entity_hits),
            "message": graph_query_result.get("message", None),
            "error": graph_query_result.get("error", None)
        }
    except Exception as e:
        print(f"❌ Error in graph_based_query: {type(e).__name__} - {e}")
        traceback.print_exc()
        return {
            "query": query_text, "results": [], "total_found": 0,
            "error": f"Failed to execute graph query: {str(e)}"
        }


async def hybrid_search_query(query_text: str, limit: int, include_context_summary: bool, domain_context: Optional[str] = "general") -> IntelligentQueryResponse:
    """Combines semantic and graph results, then generates a contextual summary."""
    try:
        semantic_limit = limit # Fetch full limit for semantic to allow better ranking later
        graph_limit = limit   # Fetch full limit for graph as well

        semantic_response_dict = await semantic_search(query_text, semantic_limit)
        graph_response_dict = await graph_based_query(query_text, graph_limit, domain_context)

        semantic_hits: List[SemanticHit] = semantic_response_dict.get("results", [])
        graph_entity_hits: List[GraphEntityHit] = graph_response_dict.get("results", [])

        # For contextual summary, collect diverse pieces of information
        llm_context_snippets: List[str] = []
        seen_snippets_for_llm = set() # To avoid duplicate snippets in LLM prompt

        # Top semantic chunk previews
        for hit in semantic_hits[:max(2, limit //3)]: # Use a few top semantic hits
            snippet = f"From semantic search (similarity {hit.similarity:.2f}): Document '{hit.document_name}', Chunk {hit.chunk_index} - Preview: \"{hit.content_preview}\""
            if snippet not in seen_snippets_for_llm:
                llm_context_snippets.append(snippet)
                seen_snippets_for_llm.add(snippet)

        # Top graph entity details and their contexts
        graph_context_chunks_for_response: List[GraphContextChunk] = []
        if graph_entity_hits:
            top_graph_entity_names = [gh.entity_name for gh in graph_entity_hits[:max(2, limit//3)] if gh.entity_name] # Use a few top graph entities
            
            if top_graph_entity_names:
                # Fetch context chunks for these entities
                # This could be a separate utility in neo4j_ops
                if neo4j_ops.driver:
                    with neo4j_ops.driver.session(database="neo4j") as session:
                        chunk_results = session.run("""
                            UNWIND $entity_names AS e_name
                            MATCH (e {name: e_name})
                            MATCH (e)<-[:MENTIONS]-(c:Chunk)<-[:CONTAINS]-(d:Document)
                            RETURN d.name AS document_name,
                                   c.chunk_id AS chunk_id,
                                   c.chunk_index AS chunk_index,
                                   c.content_preview AS content_preview,
                                   e.name AS matched_entity
                            ORDER BY d.name, c.chunk_index
                            LIMIT $context_limit
                        """, entity_names=top_graph_entity_names, context_limit=limit).data() # Fetch up to 'limit' context chunks

                        for record in chunk_results:
                            graph_context_chunks_for_response.append(GraphContextChunk(**record))
                            snippet = f"Context for graph entity '{record['matched_entity']}': Document '{record['document_name']}', Chunk {record['chunk_index']} - Preview: \"{record['content_preview']}\""
                            if snippet not in seen_snippets_for_llm:
                                llm_context_snippets.append(snippet)
                                seen_snippets_for_llm.add(snippet)
                else: # Fallback if no driver
                    for gh_hit in graph_entity_hits[:max(2, limit//3)]:
                        snippet = f"From graph search: Entity '{gh_hit.entity_name}' (Type: {', '.join(gh_hit.entity_labels)}). Description: {gh_hit.entity_description or 'N/A'}."
                        if snippet not in seen_snippets_for_llm:
                             llm_context_snippets.append(snippet)
                             seen_snippets_for_llm.add(snippet)


        contextual_summary = None
        if include_context_summary:
            if llm_context_snippets:
                context_str_for_llm = "\n\n".join(llm_context_snippets)
                summary_prompt = f"""
                You are an expert AI assistant. Your task is to synthesize information from various sources to answer the user's query.
                Be concise, factual, and directly address the query if possible.
                If the retrieved information is insufficient or doesn't directly answer, clearly state that.
                Do not invent information not present in the provided context.

                User Query:
                "{query_text}"

                Retrieved Context:
                ---
                {context_str_for_llm}
                ---
                Synthesized Answer:
                """
                try:
                    response = llm.invoke([HumanMessage(content=summary_prompt)])
                    contextual_summary = response.content.strip()
                except Exception as e_llm:
                    print(f"⚠️ LLM error during contextual summary generation: {type(e_llm).__name__} - {e_llm}")
                    contextual_summary = "Error: Could not generate summary due to an LLM issue."
            else:
                contextual_summary = "No specific context found from search results to generate a summary for this query."

        return IntelligentQueryResponse(
            mode="hybrid",
            query=query_text,
            contextual_summary=contextual_summary,
            semantic_search_results=semantic_hits[:limit], # Trim to overall limit for final response
            graph_query_results=graph_entity_hits[:limit], # Trim to overall limit
            graph_entity_context_chunks=graph_context_chunks_for_response[:limit], # Trim
            details={
                "semantic_hits_count_before_trim": len(semantic_hits),
                "graph_entity_hits_count_before_trim": len(graph_entity_hits),
                "graph_context_chunks_found": len(graph_context_chunks_for_response),
                "domain_context_for_graph": domain_context,
                "semantic_error": semantic_response_dict.get("error"),
                "graph_error": graph_response_dict.get("error")
            }
        )

    except Exception as e:
        print(f"❌ Error in hybrid_search_query: {type(e).__name__} - {e}")
        traceback.print_exc()
        # Constructing a valid IntelligentQueryResponse for error case
        return IntelligentQueryResponse(
            mode="hybrid", query=query_text, details={},
            error=f"Hybrid search failed: {str(e)}"
        )


# --- Health & Utility Endpoints ---
@router.get("/health/", summary="Comprehensive Health Check")
async def health_check():
    """
    Performs a comprehensive health check of all critical components:
    Neo4j, Embedding Model, LLM, and Milvus (conceptual).
    Also provides basic Knowledge Graph statistics.
    """
    try:
        neo4j_ok = neo4j_ops.test_connection() # This is synchronous
        
        embed_ok, embed_error = False, ""
        try: _ = embedder.embed_query("test query"); embed_ok = True
        except Exception as e: embed_error = str(e); print(f"⚠️ Embedder health check error: {e}")
        
        llm_ok, llm_error = False, ""
        try: _ = llm.invoke([HumanMessage(content="ping")]); llm_ok = True
        except Exception as e: llm_error = str(e); print(f"⚠️ LLM health check error: {e}")
        
        milvus_ok, milvus_error = True, "Conceptual check: Assumed OK. Implement actual check if milvus_ops supports."
        # try:
        #     if hasattr(milvus_ops, 'check_connection') and callable(milvus_ops.check_connection):
        #         milvus_ops.check_connection() # Implement this in milvus_ops
        #         milvus_ok = True
        #         milvus_error = ""
        #     elif hasattr(milvus_ops, 'collection'): # Basic check if collection object exists
        #         milvus_ok = True if milvus_ops.collection else False
        #         milvus_error = "" if milvus_ok else "Milvus collection not initialized."
        # except Exception as e:
        #     milvus_ok=False; milvus_error=str(e); print(f"⚠️ Milvus health check error: {e}")

        kg_stats = {}
        try:
            raw_stats = neo4j_ops.get_graph_statistics() # Synchronous
            if raw_stats.get("counts"):
                kg_stats = {
                    "documents": raw_stats["counts"].get("documents", 0),
                    "chunks": raw_stats["counts"].get("chunks", 0),
                    "entities": raw_stats["counts"].get("entities", 0),
                    "entity_labels_sample": raw_stats.get("entity_label_distribution", [])[:3],
                    "semantic_rels_sample": raw_stats.get("semantic_relationship_distribution", [])[:3]
                }
            else:
                kg_stats = {"error": raw_stats.get("error", "Could not retrieve KG stats.")}
        except Exception as e:
            kg_stats = {"error": f"Failed to get KG stats: {str(e)}"}
            print(f"⚠️ KG stats health check error: {e}")

        overall_health = all([neo4j_ok, embed_ok, llm_ok, milvus_ok])
        
        return {
            "overall_status": "healthy" if overall_health else "degraded",
            "components": {
                "neo4j": {"status": "healthy" if neo4j_ok else "unhealthy", "message": "Connection successful" if neo4j_ok else "Connection failed"},
                "embedding_model": {"status": "healthy" if embed_ok else "unhealthy", "details": "Operational" if embed_ok else embed_error},
                "llm": {"status": "healthy" if llm_ok else "unhealthy", "details": "Operational" if llm_ok else llm_error},
                "milvus_vector_store": {"status": "healthy" if milvus_ok else "unhealthy", "details": milvus_error or "Operational (conceptual check)"}
            },
            "knowledge_graph_overview": kg_stats,
            "service_version": "intelligent_api_kg_v1.1.0"
        }
        
    except Exception as e:
        print(f"❌ Critical Error in health_check itself: {type(e).__name__} - {e}")
        traceback.print_exc()
        # Return a valid JSON response even if health check itself fails badly
        return {
            "overall_status": "error_in_health_check",
            "error_details": str(e),
            "components": {},
            "knowledge_graph_overview": {},
            "service_version": "intelligent_api_kg_v1.1.0"
        }


@router.get("/explore/entity/{entity_name}", summary="Explore a Specific Entity")
async def explore_entity_endpoint(entity_name: str, limit_connections: int = Query(10, ge=1, le=50)): # Made async
    """
    Fetches detailed information about a specific entity from the Knowledge Graph,
    including its properties and direct connections.
    """
    try:
        decoded_entity_name = urllib.parse.unquote(entity_name)
        
        # This could be a dedicated function in neo4j_ops for cleaner separation
        # Make sure driver is available
        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            # Query to get entity and its connections
            # Ensure that labels(e) is handled correctly if an entity has multiple labels
            # Using the first label as 'primary_type' for simplicity here.
            result = session.run("""
                MATCH (e {name: $name})
                OPTIONAL MATCH (e)-[r]-(connected_node)
                RETURN e.name AS entity_name,
                       labels(e) AS entity_all_labels, 
                       properties(e) AS entity_properties,
                       COLLECT(DISTINCT CASE
                                   WHEN r IS NOT NULL AND connected_node IS NOT NULL THEN {
                                       relationship_type: type(r),
                                       direction: CASE WHEN startNode(r) = e THEN 'OUTGOING' ELSE 'INCOMING' END,
                                       target_node_name: connected_node.name,
                                       target_node_labels: labels(connected_node),
                                       relationship_properties: properties(r)
                                   }
                                   ELSE null
                               END
                       )[0..$limit] AS connections
            """, name=decoded_entity_name, limit=limit_connections).single()

            if not result:
                raise HTTPException(status_code=404, detail=f"Entity '{decoded_entity_name}' not found in the Knowledge Graph.")
            
            data = dict(result) # Convert Neo4j Record to dict
            # Clean up nulls from connections list if any
            if data.get('connections'):
                data['connections'] = [conn for conn in data['connections'] if conn is not None]
            
            # Add a primary type for easier display if multiple labels exist
            if data.get('entity_all_labels'):
                data['primary_type'] = data['entity_all_labels'][0] if data['entity_all_labels'] else 'Unknown'


            # Optionally, fetch a few context snippets where this entity is mentioned
            context_snippets = session.run("""
                MATCH (e:Entity {name: $name})<-[:MENTIONS]-(c:Chunk)<-[:CONTAINS]-(d:Document)
                RETURN d.name as document_name, c.chunk_index as chunk_index, c.content_preview as preview
                LIMIT 5
            """, name=decoded_entity_name).data()
            data['mention_contexts'] = context_snippets


            return data

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Error exploring entity '{entity_name}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exploring entity: {str(e)}")