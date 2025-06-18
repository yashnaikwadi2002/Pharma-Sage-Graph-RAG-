# from fastapi import APIRouter
# from backend.core.neo4j_ops import create_node, create_relationship

# router = APIRouter()

# @router.post("/graph/create_node/")
# def create_graph_node():
#     create_node("Document", {"name": "SampleDoc", "author": "AI"})
#     return {"message": "Node created"}

# @router.post("/graph/create_relationship/")
# def create_graph_rel():
#     create_relationship("Document", "name", "SampleDoc", "Author", "name", "AI", "WROTE")
#     return {"message": "Relationship created"}

# NEW CODE from CLAUDE>>><><><<><><><><><<<><<><><><><><><><><><><><><><><><><><><><><>
# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel, Field
# from typing import Dict, Any, Optional, List
# from backend.core.neo4j_ops import create_node, create_relationship
# import json

# router = APIRouter()

# # Pydantic models for request validation
# class NodeRequest(BaseModel):
#     label: str = Field(..., description="Node label (e.g., 'Document', 'Person')")
#     properties: Dict[str, Any] = Field(..., description="Node properties as key-value pairs")
    
#     class Config:
#         # Allow arbitrary types if needed
#         arbitrary_types_allowed = True
#         # Example schema
#         schema_extra = {
#             "example": {
#                 "label": "Document",
#                 "properties": {
#                     "name": "Sample Document",
#                     "author": "John Doe",
#                     "date_created": "2024-01-01"
#                 }
#             }
#         }

# class RelationshipRequest(BaseModel):
#     source_label: str = Field(..., description="Source node label")
#     source_key: str = Field(..., description="Source node property key for matching")
#     source_value: Any = Field(..., description="Source node property value for matching")
#     target_label: str = Field(..., description="Target node label")
#     target_key: str = Field(..., description="Target node property key for matching")
#     target_value: Any = Field(..., description="Target node property value for matching")
#     relationship_type: str = Field(..., description="Type of relationship (e.g., 'WROTE', 'CONTAINS')")
#     relationship_properties: Optional[Dict[str, Any]] = Field(default={}, description="Relationship properties")
    
#     class Config:
#         arbitrary_types_allowed = True
#         schema_extra = {
#             "example": {
#                 "source_label": "Person",
#                 "source_key": "name",
#                 "source_value": "John Doe",
#                 "target_label": "Document",
#                 "target_key": "name", 
#                 "target_value": "Sample Document",
#                 "relationship_type": "WROTE",
#                 "relationship_properties": {
#                     "date": "2024-01-01",
#                     "role": "author"
#                 }
#             }
#         }

# class GraphQueryRequest(BaseModel):
#     cypher_query: str = Field(..., description="Cypher query to execute")
#     parameters: Optional[Dict[str, Any]] = Field(default={}, description="Query parameters")
    
#     class Config:
#         arbitrary_types_allowed = True
#         schema_extra = {
#             "example": {
#                 "cypher_query": "MATCH (n:Document) RETURN n.name, n.author LIMIT 10",
#                 "parameters": {}
#             }
#         }

# # Original simple endpoints (keeping for backward compatibility)
# @router.post("/graph/create_node/")
# def create_graph_node():
#     """Create a sample node - kept for backward compatibility"""
#     try:
#         create_node("Document", {"name": "SampleDoc", "author": "AI"})
#         return {"message": "Sample node created successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error creating sample node: {str(e)}")

# @router.post("/graph/create_relationship/")
# def create_graph_rel():
#     """Create a sample relationship - kept for backward compatibility"""
#     try:
#         create_relationship("Document", "name", "SampleDoc", "Author", "name", "AI", "WROTE")
#         return {"message": "Sample relationship created successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error creating sample relationship: {str(e)}")

# # Enhanced endpoints with proper validation
# @router.post("/graph/nodes/")
# def create_custom_node(request: NodeRequest):
#     """Create a custom node with specified label and properties"""
#     try:
#         # Validate label name (should be a valid identifier)
#         if not request.label.replace('_', '').isalnum():
#             raise HTTPException(status_code=400, detail="Label must contain only letters, numbers, and underscores")
        
#         create_node(request.label, request.properties)
        
#         return {
#             "message": f"{request.label} node created successfully",
#             "label": request.label,
#             "properties": request.properties
#         }
        
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error creating node: {str(e)}")

# @router.post("/graph/relationships/")
# def create_custom_relationship(request: RelationshipRequest):
#     """Create a custom relationship between two nodes"""
#     try:
#         # Validate relationship type
#         if not request.relationship_type.replace('_', '').isalnum():
#             raise HTTPException(status_code=400, detail="Relationship type must contain only letters, numbers, and underscores")
        
#         create_relationship(
#             request.source_label,
#             request.source_key,
#             request.source_value,
#             request.target_label,
#             request.target_key,
#             request.target_value,
#             request.relationship_type,
#             request.relationship_properties
#         )
        
#         return {
#             "message": f"{request.relationship_type} relationship created successfully",
#             "source": f"{request.source_label}({request.source_key}: {request.source_value})",
#             "target": f"{request.target_label}({request.target_key}: {request.target_value})",
#             "relationship_type": request.relationship_type,
#             "properties": request.relationship_properties
#         }
        
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error creating relationship: {str(e)}")

# @router.post("/graph/query/")
# def execute_cypher_query(request: GraphQueryRequest):
#     """Execute a custom Cypher query"""
#     try:
#         from backend.core.neo4j_ops import driver
        
#         with driver.session() as session:
#             result = session.run(request.cypher_query, request.parameters)
            
#             # Convert result to list of dictionaries
#             records = []
#             for record in result:
#                 records.append(dict(record))
            
#             return {
#                 "query": request.cypher_query,
#                 "parameters": request.parameters,
#                 "results": records,
#                 "count": len(records)
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error executing query: {str(e)}")

# @router.get("/graph/schema/")
# def get_graph_schema():
#     """Get the current graph schema (labels and relationship types)"""
#     try:
#         from backend.core.neo4j_ops import driver
        
#         with driver.session() as session:
#             # Get all labels
#             labels_result = session.run("CALL db.labels()")
#             labels = [record["label"] for record in labels_result]
            
#             # Get all relationship types
#             rel_types_result = session.run("CALL db.relationshipTypes()")
#             relationship_types = [record["relationshipType"] for record in rel_types_result]
            
#             # Get property keys
#             prop_keys_result = session.run("CALL db.propertyKeys()")
#             property_keys = [record["propertyKey"] for record in prop_keys_result]
            
#             return {
#                 "labels": labels,
#                 "relationship_types": relationship_types,
#                 "property_keys": property_keys,
#                 "total_labels": len(labels),
#                 "total_relationship_types": len(relationship_types),
#                 "total_property_keys": len(property_keys)
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting schema: {str(e)}")

# @router.get("/graph/stats/")
# def get_graph_statistics():
#     """Get basic statistics about the graph"""
#     try:
#         from backend.core.neo4j_ops import driver
        
#         with driver.session() as session:
#             # Count nodes
#             node_count_result = session.run("MATCH (n) RETURN count(n) as node_count")
#             node_count = node_count_result.single()["node_count"]
            
#             # Count relationships
#             rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
#             rel_count = rel_count_result.single()["rel_count"]
            
#             # Count by label
#             label_counts_result = session.run("""
#                 MATCH (n) 
#                 RETURN labels(n)[0] as label, count(n) as count 
#                 ORDER BY count DESC
#             """)
#             label_counts = [{"label": record["label"], "count": record["count"]} 
#                           for record in label_counts_result]
            
#             # Count by relationship type
#             rel_type_counts_result = session.run("""
#                 MATCH ()-[r]->() 
#                 RETURN type(r) as relationship_type, count(r) as count 
#                 ORDER BY count DESC
#             """)
#             rel_type_counts = [{"type": record["relationship_type"], "count": record["count"]} 
#                              for record in rel_type_counts_result]
            
#             return {
#                 "total_nodes": node_count,
#                 "total_relationships": rel_count,
#                 "nodes_by_label": label_counts,
#                 "relationships_by_type": rel_type_counts
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

# @router.delete("/graph/clear/")
# def clear_graph():
#     """Clear all nodes and relationships from the graph - USE WITH CAUTION!"""
#     try:
#         from backend.core.neo4j_ops import driver
        
#         with driver.session() as session:
#             # Count before deletion
#             count_result = session.run("MATCH (n) RETURN count(n) as count")
#             initial_count = count_result.single()["count"]
            
#             # Delete everything
#             session.run("MATCH (n) DETACH DELETE n")
            
#             return {
#                 "message": "Graph cleared successfully",
#                 "nodes_deleted": initial_count,
#                 "warning": "All data has been permanently deleted"
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error clearing graph: {str(e)}")

# @router.get("/graph/explore/{node_label}")
# def explore_nodes_by_label(node_label: str, limit: int = 10):
#     """Explore nodes of a specific label with their relationships"""
#     try:
#         from backend.core.neo4j_ops import driver
        
#         with driver.session() as session:
#             query = f"""
#             MATCH (n:{node_label})
#             OPTIONAL MATCH (n)-[r]-(connected)
#             RETURN n, 
#                    collect({{
#                        relationship: type(r),
#                        direction: CASE 
#                            WHEN startNode(r) = n THEN 'outgoing'
#                            ELSE 'incoming'
#                        END,
#                        connected_node: labels(connected)[0],
#                        connected_properties: properties(connected)
#                    }}) as connections
#             LIMIT $limit
#             """
            
#             result = session.run(query, limit=limit)
            
#             nodes_data = []
#             for record in result:
#                 node = record["n"]
#                 connections = record["connections"]
                
#                 # Filter out null connections
#                 valid_connections = [conn for conn in connections if conn["relationship"] is not None]
                
#                 nodes_data.append({
#                     "node_properties": dict(node),
#                     "connections": valid_connections,
#                     "connection_count": len(valid_connections)
#                 })
            
#             return {
#                 "label": node_label,
#                 "nodes": nodes_data,
#                 "count": len(nodes_data),
#                 "limit": limit
#             }
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error exploring nodes: {str(e)}")

# CODE from GOOGLE<><><><><><><><<><><<><<><><><><><><><<><><><><><<<>><><<><<><><
# backend/api/routes/graph.py

from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

# Assuming neo4j_ops is the new KG-enabled version
from backend.core import neo4j_ops
import traceback
import re # For sanitizing label in explore endpoint

router = APIRouter()

# --- Pydantic Models ---
class GraphQueryRequest(BaseModel):
    cypher_query: str = Field(..., description="Cypher query to execute. Use with caution.")
    parameters: Optional[Dict[str, Any]] = Field(default={}, description="Parameters for the Cypher query.")

    class Config:
        schema_extra = {
            "example": {
                "cypher_query": "MATCH (e:PERSON) WHERE e.name STARTS WITH $letter RETURN e.name, e.description LIMIT 10",
                "parameters": {"letter": "A"}
            }
        }

class GraphSchemaResponse(BaseModel):
    labels: List[str]
    relationship_types: List[str]
    property_keys: List[str]
    # We can add more detailed schema info if db.schema.visualization() is parsed

class NodeExplorationResponse(BaseModel):
    label_explored: str
    nodes_found: List[Dict[str, Any]] # Each dict is a node with its properties and connections
    count_returned: int
    limit_applied: int


# --- Graph Utility & Admin Endpoints ---

@router.post("/admin/execute-cypher",
    tags=["Graph Admin - Advanced"],
    summary="Execute Raw Cypher Query (Admin Only)"
)
async def execute_raw_cypher_query_endpoint(request: GraphQueryRequest = Body(...)):
    """
    Allows execution of arbitrary Cypher queries against the Neo4j database.
    **WARNING:** This is a powerful endpoint intended for admin and debugging purposes only.
    Improper queries can modify or delete data, or cause performance issues.
    Ensure proper authentication and authorization are in place for this endpoint.
    """
    # TODO: Add robust authentication/authorization for this endpoint
    # from ..auth import require_admin_user
    # await require_admin_user(user_dependency)

    print(f"ℹ️ ADMIN: Executing Cypher: {request.cypher_query} with params: {request.parameters}")
    try:
        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            result = session.run(request.cypher_query, request.parameters)
            
            # Process records if it's a read query
            records = [dict(record) for record in result]
            summary = result.consume() # Important to consume for write query stats
            
            return {
                "query_executed": request.cypher_query,
                "parameters_used": request.parameters,
                "results_data": records,
                "query_summary": {
                    "query_type": summary.query_type,
                    "counters": dict(summary.counters), # Convert internal counters object
                    "plan": str(summary.plan) if summary.plan else None, # Basic plan info
                    "notifications": [vars(n) for n in summary.notifications] if summary.notifications else []
                },
                "results_count": len(records)
            }
            
    except Exception as e:
        print(f"❌ Error executing admin Cypher query: {type(e).__name__} - {e}")
        traceback.print_exc()
        # Be careful about exposing too much detail from DB errors
        error_detail = f"Error executing Cypher: {type(e).__name__}. Check server logs."
        if "constraint validation" in str(e).lower(): # Example of more specific error
            error_detail = "Constraint validation failed. Please check your query and data."
        raise HTTPException(status_code=400, detail=error_detail) # 400 for bad query, 500 for server issues


@router.get("/schema", response_model=GraphSchemaResponse, summary="Get Graph Schema Overview")
async def get_graph_schema_endpoint():
    """
    Retrieves an overview of the current graph schema, including all node labels,
    relationship types, and property keys present in the database.
    """
    try:
        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            labels_res = session.run("CALL db.labels() YIELD label RETURN label")
            labels = sorted([record["label"] for record in labels_res])
            
            rel_types_res = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            rel_types = sorted([record["relationshipType"] for record in rel_types_res])
            
            prop_keys_res = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")
            prop_keys = sorted([record["propertyKey"] for record in prop_keys_res])
            
            return GraphSchemaResponse(
                labels=labels,
                relationship_types=rel_types,
                property_keys=prop_keys
            )
    except Exception as e:
        print(f"❌ Error getting graph schema: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve graph schema.")


@router.get("/statistics", summary="Get Overall Knowledge Graph Statistics")
async def get_overall_kg_statistics_endpoint():
    """
    Retrieves comprehensive statistics about the knowledge graph, including node counts,
    entity type distributions, and relationship type distributions.
    Delegates to `neo4j_ops.get_graph_statistics()`.
    """
    try:
        # This function is already well-defined in the new neo4j_ops.py
        stats = neo4j_ops.get_graph_statistics() # Synchronous
        if "error" in stats: # Check if neo4j_ops itself returned an error structure
            raise HTTPException(status_code=500, detail=stats["error"])
        return stats # neo4j_ops.get_graph_statistics already returns a dict
    except Exception as e:
        print(f"❌ Error getting overall graph statistics: {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to retrieve overall graph statistics.")


@router.get("/explore/nodes-by-label/{node_label}", response_model=NodeExplorationResponse, summary="Explore Nodes by Label")
async def explore_nodes_by_label_endpoint(
    node_label: str,
    limit: int = Query(10, ge=1, le=50, description="Number of nodes to return")
):
    """
    Explores nodes of a specific label and shows their properties and a sample of their direct connections.
    Useful for understanding the structure and content associated with a particular node type.
    """
    try:
        # Sanitize node_label to ensure it's a valid Neo4j label (alphanumeric + underscore)
        # This is important because it's used directly in the Cypher query.
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", node_label):
            raise HTTPException(status_code=400, detail="Invalid node label format. Labels must be alphanumeric and can include underscores, starting with a letter or underscore.")

        if not neo4j_ops.driver:
            raise HTTPException(status_code=503, detail="Neo4j driver not initialized.")

        with neo4j_ops.driver.session(database="neo4j") as session:
            # Using f-string for label is generally safe IF SANITIZED. Parameterization for labels is tricky.
            # The $node_label parameter cannot be used directly as a label in Cypher like `MATCH (n:$node_label)`.
            # Alternative: `MATCH (n) WHERE $node_label IN labels(n)` but less performant.
            query = f"""
            MATCH (n:{node_label})
            WITH n LIMIT $limit // Apply limit early for performance
            OPTIONAL MATCH (n)-[r]-(connected_node)
            RETURN n AS node,
                   COLLECT(DISTINCT CASE
                       WHEN r IS NOT NULL AND connected_node IS NOT NULL THEN {{
                           relationship_type: type(r),
                           direction: CASE WHEN startNode(r) = n THEN 'OUTGOING' ELSE 'INCOMING' END,
                           target_node_name: connected_node.name, // Assuming connected nodes have a 'name'
                           target_node_labels: labels(connected_node)
                       }}
                       ELSE null
                   END
                   )[0..5] AS connections // Limit connections per node in result
            """
            
            result = session.run(query, limit=limit) # Removed node_label from params as it's in f-string
            
            nodes_data = []
            for record in result:
                node_properties = dict(record["node"])
                connections_raw = record["connections"]
                # Filter out nulls from connections if any (e.g., node has no relationships)
                valid_connections = [conn for conn in connections_raw if conn is not None]
                
                nodes_data.append({
                    "node_id": record["node"].element_id, # Neo4j internal ID
                    "node_labels": list(record["node"].labels),
                    "node_properties": node_properties,
                    "connections_sample": valid_connections
                })
            
        return NodeExplorationResponse(
            label_explored=node_label,
            nodes_found=nodes_data,
            count_returned=len(nodes_data),
            limit_applied=limit
        )
            
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"❌ Error exploring nodes by label '{node_label}': {type(e).__name__} - {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error exploring nodes by label: {str(e)}")

# The generic /graph/create_node, /graph/create_relationship, /graph/nodes, /graph/relationships
# from your "NEW CODE from CLAUDE" are COMMENTED OUT.
# Reason: They rely on generic `create_node` and `create_relationship` functions that
# are no longer in our KG-focused `neo4j_ops.py`.
# Adding arbitrary nodes/relationships outside the KG pipeline can lead to inconsistency.
# If such functionality is truly needed for admin/debug, it would require
# careful design to ensure compatibility or clear separation.

# The `/graph/clear` endpoint is also removed as a more contextualized version
# exists in `documents.py` at `/api/documents/admin/clear-all-data`.
# Avoid duplicating highly destructive endpoints.