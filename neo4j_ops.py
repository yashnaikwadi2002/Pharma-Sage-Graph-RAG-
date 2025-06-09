# from neo4j import GraphDatabase
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Neo4j connection settings for free tier
# NEO4J_URI = os.getenv("NEO4J_URI", "bolt+ssc://85759ac7.databases.neo4j.io")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Y4RrATrfzCwD1JXgXnyfBxX8T3MUW4OJtbsZypVhBjU")

# # Create driver with connection pool settings
# driver = GraphDatabase.driver(
#     NEO4J_URI, 
#     auth=(NEO4J_USER, NEO4J_PASSWORD),
#     max_connection_lifetime=30 * 60,  # 30 minutes
#     max_connection_pool_size=50,
#     connection_acquisition_timeout=60  # 60 seconds
# )

# def test_connection():
#     """Test Neo4j connection with verify_connectivity"""
#     try:
#         # Use verify_connectivity like in your working test
#         driver.verify_connectivity()
#         print("✅ Neo4j connection successful")
#         return True
#     except Exception as e:
#         print(f"❌ Neo4j connection failed: {str(e)}")
#         return False


# def create_node(label, properties):
#     """Create a single node with dynamic label and properties"""
#     if not label.isidentifier():
#         raise ValueError("Invalid label provided.")
    
#     try:
#         query = f"CREATE (n:{label}) SET n += $props"
#         with driver.session() as session:
#             session.run(query, props=properties)
#             print(f"✅ Created {label} node")
#     except Exception as e:
#         print(f"❌ Error creating {label} node: {str(e)}")
#         raise


# def create_relationship(node1_label, node1_key, node1_value,
#                         node2_label, node2_key, node2_value,
#                         rel_type, rel_props={}):
#     """Create a relationship between two nodes"""
#     try:
#         query = (
#             f"MATCH (a:{node1_label}), (b:{node2_label}) "
#             f"WHERE a.{node1_key} = $value1 AND b.{node2_key} = $value2 "
#             f"CREATE (a)-[r:{rel_type} $props]->(b)"
#         )
#         with driver.session() as session:
#             result = session.run(query, value1=node1_value, value2=node2_value, props=rel_props)
#             print(f"✅ Created {rel_type} relationship")
#     except Exception as e:
#         print(f"❌ Error creating relationship: {str(e)}")
#         raise


# def create_graph(chunks, filename):
#     """
#     Create a knowledge graph from document chunks.
#     Creates nodes for document and its chunks, and establishes relationships.
#     """
#     try:
#         print(f"🔗 Starting graph creation for {filename}")
        
#         with driver.session() as session:
#             # Create document node
#             print(f"📄 Creating document node for {filename}")
#             session.run(
#                 "CREATE (d:Document) SET d.name = $name, d.type = $type, d.chunk_count = $chunk_count",
#                 name=filename, type="Document", chunk_count=len(chunks)
#             )

#             # Create chunk nodes and relationships
#             for i, chunk_text in enumerate(chunks):
#                 chunk_id = f"{filename}_chunk_{i}"
                
#                 if i % 10 == 0:  # Progress indicator
#                     print(f"📊 Processing chunk {i+1}/{len(chunks)}")

#                 # Create chunk node
#                 session.run(
#                     "CREATE (c:Chunk) SET c.chunk_id = $chunk_id, c.content = $content, c.chunk_index = $chunk_index, c.word_count = $word_count",
#                     chunk_id=chunk_id,
#                     content=chunk_text[:500],  # Truncate long content
#                     chunk_index=i,
#                     word_count=len(chunk_text.split())
#                 )

#                 # Link document to chunk
#                 session.run(
#                     """
#                     MATCH (d:Document {name: $doc_name}), (c:Chunk {chunk_id: $chunk_id})
#                     CREATE (d)-[:CONTAINS]->(c)
#                     """,
#                     doc_name=filename,
#                     chunk_id=chunk_id
#                 )

#                 # Link chunks sequentially
#                 if i > 0:
#                     prev_chunk_id = f"{filename}_chunk_{i-1}"
#                     session.run(
#                         """
#                         MATCH (c1:Chunk {chunk_id: $prev_chunk_id}), (c2:Chunk {chunk_id: $curr_chunk_id})
#                         CREATE (c1)-[:NEXT]->(c2)
#                         """,
#                         prev_chunk_id=prev_chunk_id,
#                         curr_chunk_id=chunk_id
#                     )

#             print(f"✅ Successfully created graph for {filename} with {len(chunks)} chunks")

#     except Exception as e:
#         print(f"❌ Error creating graph for {filename}: {str(e)}")
#         raise


# def delete_document_graph(filename):
#     """
#     Delete all nodes and relationships for a specific document
#     """
#     try:
#         print(f"🗑️ Deleting graph for {filename}")
#         with driver.session() as session:
#             # First get count of nodes to be deleted
#             result = session.run(
#                 """
#                 MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)
#                 RETURN count(c) as chunk_count
#                 """,
#                 filename=filename
#             )
#             count_record = result.single()
#             chunk_count = count_record["chunk_count"] if count_record else 0
            
#             # Delete the nodes and relationships
#             session.run(
#                 """
#                 MATCH (d:Document {name: $filename})-[:CONTAINS]->(c:Chunk)
#                 DETACH DELETE c, d
#                 """,
#                 filename=filename
#             )
#             print(f"✅ Deleted graph for {filename} ({chunk_count} chunks)")
            
#     except Exception as e:
#         print(f"❌ Error deleting graph for {filename}: {str(e)}")
#         raise


# def get_document_stats():
#     """Get statistics about documents in the graph"""
#     try:
#         with driver.session() as session:
#             result = session.run(
#                 """
#                 MATCH (d:Document)
#                 OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
#                 RETURN d.name as document, count(c) as chunk_count
#                 ORDER BY d.name
#                 """
#             )
            
#             stats = []
#             for record in result:
#                 stats.append({
#                     "document": record["document"],
#                     "chunk_count": record["chunk_count"]
#                 })
            
#             return stats
            
#     except Exception as e:
#         print(f"❌ Error getting document stats: {str(e)}")
#         return []


# def close_connection():
#     """Close the Neo4j driver connection"""
#     try:
#         driver.close()
#         print("✅ Neo4j connection closed")
#     except Exception as e:
#         print(f"❌ Error closing Neo4j connection: {str(e)}")


# # Test the connection when module is imported
# if __name__ == "__main__":
#     print("Testing Neo4j connection...")
#     if test_connection():
#         print("Connection test passed!")
        
#         # Optional: Show current document stats
#         stats = get_document_stats()
#         if stats:
#             print("\nCurrent documents in graph:")
#             for stat in stats:
#                 print(f"  - {stat['document']}: {stat['chunk_count']} chunks")
#         else:
#             print("No documents found in graph")
#     else:
#         print("Connection test failed!")
    
#     close_connection()


# NEW CODE>>> <<<><><><><><><><><<><<><><><><><><><<><><><><><><><><><><><><>

# backend/core/neo4j_ops.py

from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import re
import json
from typing import List, Dict, Tuple # Tuple was in your new code, keeping it
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

load_dotenv()

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "bolt+ssc://85759ac7.databases.neo4j.io") # Replace with your actual URI if not default
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Y4RrATrfzCwD1JXgXnyfBxX8T3MUW4OJtbsZypVhBjU") # Replace with your actual password

# LLM and API Key Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

# Initialize LLM for entity extraction
# Make sure the API key is correctly loaded for the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Using flash for speed and cost
    google_api_key=GEMINI_API_KEY,
    temperature=0.1, # Low temperature for more deterministic extraction
    max_tokens=2048, # Increased for potentially larger JSON outputs from LLM
    timeout=120,     # Increased timeout for LLM calls
    max_retries=2
)

# Create driver with connection pool settings
try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=30 * 60,  # 30 minutes
        max_connection_pool_size=50,
        connection_acquisition_timeout=60  # 60 seconds
    )
except Exception as e:
    print(f"❌ Critical: Failed to create Neo4j driver: {str(e)}")
    # Depending on your application structure, you might want to exit or raise a more specific exception
    raise

def test_connection():
    """Test Neo4j connection"""
    if driver is None:
        print("❌ Neo4j driver is not initialized.")
        return False
    try:
        driver.verify_connectivity()
        print("✅ Neo4j connection successful")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {str(e)}")
        return False

def extract_entities_and_relationships(text: str, domain: str = "general") -> Dict:
    """
    Extract entities and relationships from text using LLM.
    Domain can be specified (e.g., "biomedical", "financial") to tailor extraction.
    """
    try:
        # Reduced max entities/relationships per chunk to manage LLM token limits and Neo4j write performance
        # Also made entity/relationship types more generic for broader applicability,
        # but this can be customized per domain.
        prompt = f"""
        You are an expert knowledge graph builder. Analyze the following text from a {domain} document.
        Extract key entities and their relationships.
        Return ONLY a valid JSON object with this exact structure:
        {{
            "entities": [
                {{"name": "entity_name", "type": "ENTITY_TYPE", "description": "brief description (max 30 words)"}}
            ],
            "relationships": [
                {{"source": "entity1_name", "target": "entity2_name", "type": "RELATIONSHIP_TYPE", "description": "relationship context (max 30 words)"}}
            ]
        }}

        Guidelines:
        - Entity Types: Use common types like PERSON, ORGANIZATION, LOCATION, PRODUCT, DATE, EVENT, CONCEPT, TECHNOLOGY, MEDICAL_CONDITION, DRUG, GENE, PROTEIN. Be specific if possible.
        - Relationship Types: Use descriptive verbs or short phrases like IS_A, PART_OF, LOCATED_IN, WORKS_FOR, INTERACTS_WITH, CAUSES, TREATS, ASSOCIATED_WITH, PRODUCES, USES, OCCURRED_ON.
        - Max 7 entities per text snippet.
        - Max 10 relationships per text snippet.
        - Entity names should be concise and normalized (e.g., "AI" instead of "Artificial intelligence").
        - Only create relationships between entities you've extracted in this snippet.
        - Descriptions should be very brief and informative.

        Text:
        ---
        {text[:2000]}
        ---
        JSON Output:
        """ # text[:2000] to limit input token size for the LLM

        response = llm.invoke([HumanMessage(content=prompt)])
        response_text = response.content.strip()

        # Clean the response to extract JSON
        response_text = re.sub(r'```json\s*', '', response_text)
        response_text = re.sub(r'```\s*$', '', response_text)
        response_text = response_text.replace('\\n', '\n') # Handle escaped newlines if any

        try:
            # Try to find JSON in the response more robustly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
            else: # Fallback if regex fails but string might be valid JSON
                result = json.loads(response_text)
        except json.JSONDecodeError as je:
            print(f"⚠️ LLM response was not valid JSON. Error: {je}. Response text: '{response_text[:300]}...'")
            return {"entities": [], "relationships": []}


        # Validate and clean the result
        entities = result.get("entities", [])[:7]  # Max 7 entities
        relationships = result.get("relationships", [])[:10]  # Max 10 relationships

        cleaned_entities = []
        for entity in entities:
            name = str(entity.get("name", "")).strip()
            entity_type = str(entity.get("type", "CONCEPT")).upper().replace(" ", "_")
            description = str(entity.get("description", ""))[:150] # Limit description length

            if name and len(name) > 1 and len(name) < 150:  # Ensure meaningful entity names
                cleaned_entities.append({
                    "name": name,
                    "type": entity_type if entity_type.isidentifier() else "CONCEPT",
                    "description": description
                })
        
        cleaned_relationships = []
        entity_names_in_snippet = {e["name"] for e in cleaned_entities}

        for rel in relationships:
            source = str(rel.get("source", "")).strip()
            target = str(rel.get("target", "")).strip()
            rel_type = str(rel.get("type", "RELATED_TO")).upper().replace(" ", "_")
            description = str(rel.get("description", ""))[:150]

            # Only keep relationships between entities extracted in this specific snippet and ensure type is valid
            if source in entity_names_in_snippet and target in entity_names_in_snippet and source != target:
                # Sanitize relationship type for Cypher (ensure it's a valid identifier)
                rel_type_sanitized = re.sub(r'[^A-Z0-9_]', '', rel_type) # Allow numbers as well
                if not rel_type_sanitized: rel_type_sanitized = "RELATED_TO"
                if not rel_type_sanitized[0].isalpha(): rel_type_sanitized = "R_" + rel_type_sanitized # Must start with letter

                cleaned_relationships.append({
                    "source": source,
                    "target": target,
                    "type": rel_type_sanitized,
                    "description": description
                })
        
        return {"entities": cleaned_entities, "relationships": cleaned_relationships}

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error in extract_entities_and_relationships: {str(e)}")
        print(f"LLM Response was: {response_text[:200]}...")
        return {"entities": [], "relationships": []}
    except Exception as e:
        print(f"❌ Error in extract_entities_and_relationships: {str(e)} (Type: {type(e).__name__})")
        return {"entities": [], "relationships": []}


def create_entity_node(tx, entity: Dict, source_document_filename: str):
    """Create or update an entity node within a transaction."""
    entity_name = entity.get("name")
    entity_type = entity.get("type", "CONCEPT") # Default type
    description = entity.get("description", "")

    # Basic validation
    if not entity_name or not isinstance(entity_name, str) or len(entity_name) < 2 or len(entity_name) > 250: # Max length for name
        print(f"⚠️ Skipping invalid entity name: {entity_name}")
        return None
    if not entity_type.isidentifier(): # Ensure entity type is a valid Cypher label
        print(f"⚠️ Skipping invalid entity type: {entity_type} for entity {entity_name}")
        return None


    # Using MERGE on name and type for better uniqueness if desired, or just name.
    # Here, merging on name only, and type/description can be updated.
    # Storing source_documents as a list property.
    query = f"""
    MERGE (e:{entity_type} {{name: $name}})
    ON CREATE SET
        e.description = $description,
        e.source_documents = [$source_doc],
        e.created_at = datetime(),
        e.updated_at = datetime(),
        e.mention_count = 1
    ON MATCH SET
        e.description = CASE WHEN e.description IS NULL OR e.description = '' THEN $description ELSE e.description END,
        e.source_documents = CASE
            WHEN NOT $source_doc IN e.source_documents THEN e.source_documents + $source_doc
            ELSE e.source_documents
        END,
        e.updated_at = datetime(),
        e.mention_count = e.mention_count + 1
    RETURN elementId(e) AS id, e.name AS name, e.type AS type
    """
    try:
        result = tx.run(query,
                        name=entity_name,
                        description=description,
                        source_doc=source_document_filename)
        record = result.single()
        return record # Returns some info about the created/merged node
    except Exception as e:
        print(f"❌ Error creating entity node '{entity_name}' of type '{entity_type}': {str(e)}")
        # Log detailed error, potentially with query and params for debugging
        return None

def create_entity_relationship(tx, relationship: Dict, source_document_filename: str):
    """Create a relationship between two entities within a transaction."""
    source_name = relationship.get("source")
    target_name = relationship.get("target")
    rel_type = relationship.get("type", "RELATED_TO")
    description = relationship.get("description", "")

    if not source_name or not target_name or source_name == target_name:
        return None
    if not rel_type.isidentifier() or not rel_type[0].isalpha(): # Ensure rel_type is valid
        print(f"⚠️ Skipping invalid relationship type: {rel_type} between {source_name} and {target_name}")
        return None

    # MERGE relationship to avoid duplicates, update properties on match.
    # Note: This MERGE matches on the relationship type AND endpoints.
    # If you want multiple relationships of the same type between same nodes, use CREATE.
    query = f"""
    MATCH (a {{name: $source_name}}), (b {{name: $target_name}})
    MERGE (a)-[r:{rel_type}]->(b)
    ON CREATE SET
        r.description = $description,
        r.source_documents = [$source_doc],
        r.created_at = datetime(),
        r.updated_at = datetime(),
        r.weight = 1
    ON MATCH SET
        r.description = CASE WHEN r.description IS NULL OR r.description = '' THEN $description ELSE r.description END,
        r.source_documents = CASE
            WHEN NOT $source_doc IN r.source_documents THEN r.source_documents + $source_doc
            ELSE r.source_documents
        END,
        r.updated_at = datetime(),
        r.weight = r.weight + 1
    RETURN elementId(r) AS id
    """
    try:
        result = tx.run(query,
                        source_name=source_name,
                        target_name=target_name,
                        description=description,
                        source_doc=source_document_filename)
        return result.single()
    except Exception as e:
        print(f"❌ Error creating relationship {source_name}-[{rel_type}]->{target_name}: {str(e)}")
        return None

def create_knowledge_graph(chunks: List[str], filename: str, domain: str = "general"):
    """
    Create a knowledge graph from document chunks.
    - Creates Document and Chunk nodes.
    - Extracts Entities and Relationships from each chunk using LLM.
    - Creates Entity nodes and relationships between them.
    - Links Chunks to the Entities they MENTION.
    """
    print(f"🔗 Starting knowledge graph creation for '{filename}' (domain: {domain})")
    total_chunks = len(chunks)
    doc_node_created = False
    entities_created_count = 0
    relationships_created_count = 0
    mentions_created_count = 0
    chunks_processed_for_entities = 0

    with driver.session(database="neo4j") as session: # Specify database if not default
        try:
            # Create Document node
            doc_result = session.execute_write(
                lambda tx: tx.run("""
                    MERGE (d:Document {name: $name})
                    ON CREATE SET d.type = 'Document', d.chunk_count = $chunk_count, d.created_at = datetime(), d.status = 'PendingEntityExtraction'
                    ON MATCH SET d.chunk_count = $chunk_count, d.updated_at = datetime(), d.status = 'PendingEntityExtraction'
                    RETURN elementId(d) AS id
                """, name=filename, chunk_count=total_chunks).single()
            )
            if doc_result and doc_result["id"]:
                doc_node_created = True
                print(f"📄 Document node for '{filename}' ensured.")
            else:
                print(f"⚠️ Failed to create/merge Document node for '{filename}'. Aborting graph creation for this file.")
                return


            # Process chunks
            for i, chunk_text in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                print(f"📊 Processing chunk {i + 1}/{total_chunks} for '{filename}' (ID: {chunk_id})")

                # Create Chunk node and link to Document
                # Using a single transaction for chunk creation and linking
                def create_chunk_and_link(tx, doc_name, c_id, c_text, c_idx):
                    tx.run("""
                        MATCH (d:Document {name: $doc_name})
                        CREATE (c:Chunk {
                            chunk_id: $c_id,
                            content_preview: $c_content_preview,
                            chunk_index: $c_idx,
                            word_count: $c_word_count,
                            char_count: $c_char_count,
                            created_at: datetime()
                        })
                        MERGE (d)-[r:CONTAINS {sequence: $c_idx}]->(c)
                        RETURN elementId(c) AS id
                        """, doc_name=doc_name, c_id=c_id,
                           c_content_preview=chunk_text[:250] + ('...' if len(chunk_text) > 250 else ''), # Store only preview
                           c_idx=i,
                           c_word_count=len(chunk_text.split()),
                           c_char_count=len(chunk_text)
                    )
                    # Link chunks sequentially
                    if i > 0:
                        prev_chunk_id = f"{filename}_chunk_{i-1}"
                        tx.run("""
                            MATCH (c1:Chunk {chunk_id: $prev_id}), (c2:Chunk {chunk_id: $curr_id})
                            MERGE (c1)-[:NEXT {sequence: $curr_idx}]->(c2)
                            """, prev_id=prev_chunk_id, curr_id=chunk_id, curr_idx=i
                        )
                    return True

                session.execute_write(create_chunk_and_link, filename, chunk_id, chunk_text, i)

                # Entity and Relationship Extraction for substantial chunks
                if len(chunk_text.strip()) > 50: # Minimum length for meaningful extraction
                    extracted_data = extract_entities_and_relationships(chunk_text, domain=domain)
                    if not extracted_data or (not extracted_data.get("entities") and not extracted_data.get("relationships")):
                        print(f"  -> No entities/relationships extracted from chunk {i+1}.")
                        continue
                    
                    chunks_processed_for_entities += 1
                    
                    # Create Entity nodes and MENTIONS relationships
                    # Batch these operations within a transaction for performance
                    def process_entities_for_chunk(tx, entities_list, c_id, doc_filename):
                        chunk_entities_created = 0
                        chunk_mentions_created = 0
                        for entity_data in entities_list:
                            created_entity_info = create_entity_node(tx, entity_data, doc_filename)
                            if created_entity_info:
                                chunk_entities_created +=1
                                # Link Chunk to Entity it MENTIONS
                                rel_result = tx.run("""
                                    MATCH (c:Chunk {chunk_id: $c_id}), (e {name: $e_name}) WHERE elementId(e) = $e_id
                                    MERGE (c)-[m:MENTIONS]->(e)
                                    ON CREATE SET m.created_at = datetime(), m.confidence = 1.0
                                    ON MATCH SET m.confidence = m.confidence + 0.1, m.updated_at = datetime()
                                    RETURN elementId(m) AS id
                                """, c_id=c_id, e_name=created_entity_info["name"], e_id=created_entity_info["id"])
                                if rel_result.single():
                                    chunk_mentions_created +=1
                        return chunk_entities_created, chunk_mentions_created

                    num_ent, num_ment = session.execute_write(process_entities_for_chunk, extracted_data.get("entities", []), chunk_id, filename)
                    entities_created_count += num_ent
                    mentions_created_count += num_ment
                    if num_ent > 0: print(f"  -> Created/Updated {num_ent} entities and {num_ment} MENTIONS links for chunk {i+1}.")

                    # Create Relationships between Entities
                    # Batch these operations within a transaction
                    def process_relationships_for_chunk(tx, relationships_list, doc_filename):
                        chunk_rels_created = 0
                        for rel_data in relationships_list:
                            if create_entity_relationship(tx, rel_data, doc_filename):
                                chunk_rels_created +=1
                        return chunk_rels_created

                    num_rels = session.execute_write(process_relationships_for_chunk, extracted_data.get("relationships", []), filename)
                    relationships_created_count += num_rels
                    if num_rels > 0: print(f"  -> Created/Updated {num_rels} relationships between entities from chunk {i+1}.")
                else:
                    print(f"  -> Chunk {i+1} too short, skipping entity extraction.")


            # Update Document node with final stats
            session.execute_write(
                lambda tx: tx.run("""
                    MATCH (d:Document {name: $name})
                    SET d.entity_extraction_completed_at = datetime(),
                        d.status = 'Completed',
                        d.total_entities_processed = coalesce(d.total_entities_processed, 0) + $entities_count,
                        d.total_relationships_processed = coalesce(d.total_relationships_processed, 0) + $rels_count,
                        d.total_mentions_processed = coalesce(d.total_mentions_processed, 0) + $mentions_count,
                        d.chunks_processed_for_entities = $chunks_for_entities
                    RETURN d.name
                """, name=filename, entities_count=entities_created_count,
                   rels_count=relationships_created_count, mentions_count=mentions_created_count,
                   chunks_for_entities=chunks_processed_for_entities).single()
            )
            print(f"✅ Knowledge graph processing completed for '{filename}'.")
            print(f"   Total entities created/updated: {entities_created_count}")
            print(f"   Total relationships created/updated: {relationships_created_count}")
            print(f"   Total MENTIONS links created/updated: {mentions_created_count}")
            print(f"   Chunks processed for entity extraction: {chunks_processed_for_entities}/{total_chunks}")

        except Exception as e:
            print(f"❌ Error during knowledge graph creation for '{filename}': {str(e)} (Type: {type(e).__name__})")
            # Optionally update document status to 'Failed'
            session.execute_write(
                lambda tx: tx.run("""
                    MATCH (d:Document {name: $name})
                    SET d.status = 'Failed', d.error_message = $error
                """, name=filename, error=str(e)).single()
            )
            raise # Re-raise the exception if you want the caller to handle it

def query_knowledge_graph(query_text: str, limit: int = 10, domain_context: str = "general"):
    """
    Query the knowledge graph based on a natural language query.
    This is a placeholder for a more sophisticated query function.
    A true NL to Cypher would involve more complex LLM prompting and Cypher generation.
    This version will find entities matching terms in the query.
    """
    print(f"🔎 Querying KG with: '{query_text}', limit: {limit}, domain: {domain_context}")
    # For now, let's use a simple entity search based on the query text
    # A more advanced version would use the LLM to parse the query_text into structured graph query components
    
    # Simple keyword extraction (can be improved with LLM)
    keywords = [kw.strip().lower() for kw in query_text.split() if len(kw.strip()) > 2]
    if not keywords:
        return {"query": query_text, "results": [], "message": "No suitable keywords extracted from query."}

    cypher_parts = []
    params = {}
    for i, kw in enumerate(keywords[:5]): # Limit number of keywords to use in query
        param_name = f"kw{i}"
        # Search in entity name or description
        cypher_parts.append(f"(toLower(e.name) CONTAINS ${param_name} OR toLower(e.description) CONTAINS ${param_name})")
        params[param_name] = kw
    
    if not cypher_parts:
        return {"query": query_text, "results": [], "message": "No parts for cypher query."}

    # Construct the Cypher query
    # This query finds entities and their immediate connections.
    full_cypher_query = f"""
    MATCH (e)
    WHERE {' OR '.join(cypher_parts)}
    OPTIONAL MATCH (e)-[r]-(related_e)
    WITH e, type(r) as rel_type, PROPERTIES(r) as rel_props, related_e
    ORDER BY e.mention_count DESC, size(e.name) ASC
    LIMIT {limit * 5} // Fetch more initially to collect distinct entities up to limit
    RETURN e.name AS entity_name, 
           labels(e) AS entity_labels, 
           e.description AS entity_description, 
           e.mention_count AS entity_mentions,
           COLLECT(DISTINCT {{
               relationship_type: rel_type,
               related_entity_name: related_e.name,
               related_entity_labels: labels(related_e),
               related_entity_description: related_e.description,
               relationship_properties: rel_props
           }}) AS connections
    LIMIT {limit}
    """
    # The above query collects connections for each matched 'e'. If 'e' has no connections, 'connections' will be an empty list or list with nulls.
    # Need to handle cases where related_e or r is null carefully in the collection.

    refined_cypher_query = f"""
    MATCH (e)
    WHERE {' OR '.join(cypher_parts)}
    WITH e
    ORDER BY e.mention_count DESC, size(e.name) ASC
    LIMIT $main_limit
    OPTIONAL MATCH (e)-[r]-(related_e)
    RETURN e.name AS entity_name,
           labels(e) AS entity_labels,
           e.description AS entity_description,
           e.mention_count AS entity_mentions,
           COLLECT(DISTINCT CASE
                       WHEN r IS NOT NULL AND related_e IS NOT NULL THEN {{
                           relationship_type: type(r),
                           direction: CASE WHEN startNode(r) = e THEN 'OUTGOING' ELSE 'INCOMING' END,
                           target_node_name: related_e.name,
                           target_node_labels: labels(related_e)
                       }}
                       ELSE null
                   END
           )[0..5] AS direct_connections // Limit connections per entity in result
    """
    params['main_limit'] = limit

    results_list = []
    try:
        with driver.session(database="neo4j") as session:
            records = session.run(refined_cypher_query, **params)
            for record in records:
                data = record.data()
                # Filter out null connections if any resulted from the CASE
                if data.get('direct_connections'):
                    data['direct_connections'] = [conn for conn in data['direct_connections'] if conn is not None]
                results_list.append(data)
        print(f"  -> Found {len(results_list)} results for query.")
        return {"query": query_text, "results": results_list, "cypher_executed": refined_cypher_query}
    except Exception as e:
        print(f"❌ Error querying knowledge graph: {str(e)}")
        return {"query": query_text, "results": [], "error": str(e), "cypher_executed": refined_cypher_query}


def delete_document_graph(filename: str):
    """
    Delete a document, its chunks, and MENTIONS relationships.
    Optionally, could add logic to delete entities if they are no longer mentioned by any chunk.
    """
    print(f"🗑️ Attempting to delete graph data for document: '{filename}'")
    with driver.session(database="neo4j") as session:
        # Get counts before deletion for logging
        summary = session.run("""
            MATCH (d:Document {name: $filename})
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            OPTIONAL MATCH (c)-[m:MENTIONS]->(e)
            RETURN d.name AS doc_name, count(DISTINCT c) AS chunk_count, count(DISTINCT m) AS mentions_count
        """, filename=filename).single()

        if not summary or not summary["doc_name"]:
            print(f"  -> Document '{filename}' not found. No deletion performed.")
            return {"message": f"Document '{filename}' not found."}

        print(f"  -> Found document '{summary['doc_name']}' with {summary['chunk_count']} chunks and {summary['mentions_count']} MENTIONS links.")

        # Detach and delete Chunks and the Document node
        # MENTIONS relationships from these chunks will be deleted due to DETACH on chunks
        session.execute_write(
            lambda tx: tx.run("""
                MATCH (d:Document {name: $filename})
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                DETACH DELETE c, d
            """, filename=filename)
        )
        print(f"  -> Document node and its chunks (with their MENTIONS relationships) deleted for '{filename}'.")

        # Optional: Clean up orphaned entities (entities no longer mentioned by any chunk from any document)
        # This can be resource-intensive if run frequently on large graphs.
        # Consider running this as a periodic maintenance task.
        # orphaned_check_result = session.execute_write(
        #     lambda tx: tx.run("""
        #         MATCH (e) WHERE size(labels(e)) > 0 AND NOT (e:Document OR e:Chunk) // Ensure it's an entity node
        #         AND NOT EXISTS ((:Chunk)-[:MENTIONS]->(e))
        #         WITH e LIMIT 1000 // Process in batches to avoid long transactions
        #         DETACH DELETE e
        #         RETURN count(e) AS orphaned_deleted_count
        #     """).single()
        # )
        # if orphaned_check_result and orphaned_check_result["orphaned_deleted_count"] > 0:
        #     print(f"  -> Cleaned up {orphaned_check_result['orphaned_deleted_count']} orphaned entities.")
        
        return {
            "message": f"Successfully deleted graph data for document '{filename}'.",
            "chunks_deleted": summary["chunk_count"],
            "mentions_links_deleted_with_chunks": summary["mentions_count"]
            # "orphaned_entities_cleaned": orphaned_check_result["orphaned_deleted_count"] if orphaned_check_result else 0
        }

# def get_document_stats():
#     """Get statistics about documents in the graph"""
#     try:
#         with driver.session(database="neo4j") as session:
#             result = session.run("""
#                 MATCH (d:Document)
#                 OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
#                 OPTIONAL MATCH (c)-[m:MENTIONS]->(e)
#                 RETURN d.name AS document,
#                        d.status AS status,
#                        d.chunk_count AS total_chunks_in_doc,
#                        count(DISTINCT c) AS chunks_in_graph, // Chunks actually in graph for this doc
#                        count(DISTINCT e) AS unique_entities_mentioned,
#                        d.created_at AS created_at,
#                        d.entity_extraction_completed_at AS completed_at,
#                        d.error_message AS error
#                 ORDER BY d.name
#             """)
#             stats = [record.data() for record in result]
#             return stats
#     except Exception as e:
#         print(f"❌ Error getting document stats: {str(e)}")
#         return []

# In backend/core/neo4j_ops.py

def get_document_stats():
    """Get statistics about documents in the graph"""
    try:
        if not driver:
            print("❌ Neo4j driver not initialized in get_document_stats.")
            return []
        with driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                OPTIONAL MATCH (c)-[m:MENTIONS]->(e)
                RETURN d.name AS document,
                       d.status AS status,
                       d.chunk_count AS total_chunks_in_doc, // Total chunks as per original count
                       count(DISTINCT c) AS chunks_actually_in_graph, // Chunks physically in graph for this doc
                       d.chunks_processed_for_entities AS chunks_processed_for_kg, // <--- ADDED THIS
                       count(DISTINCT e) AS unique_entities_mentioned, // Entities mentioned by chunks of this doc
                       d.created_at AS created_at,
                       d.entity_extraction_completed_at AS kg_processing_completed_at, // Renamed alias for clarity
                       d.error_message AS error_message // Renamed alias for clarity
                ORDER BY d.name
            """)
            stats = [record.data() for record in result]
            return stats
    except Exception as e:
        print(f"❌ Error getting document stats: {str(e)}")
        return []

def get_graph_statistics():
    """Get comprehensive graph statistics"""
    stats = {}
    try:
        with driver.session(database="neo4j") as session:
            stats["node_labels"] = [record.data() for record in session.run("CALL db.labels()")]
            stats["relationship_types"] = [record.data() for record in session.run("CALL db.relationshipTypes()")]
            
            doc_count = session.run("MATCH (d:Document) RETURN count(d) AS count").single()["count"]
            chunk_count = session.run("MATCH (c:Chunk) RETURN count(c) AS count").single()["count"]
            entity_count = session.run("MATCH (e) WHERE size(labels(e)) > 0 AND NOT (e:Document OR e:Chunk) RETURN count(e) AS count").single()["count"]
            
            stats["counts"] = {
                "documents": doc_count,
                "chunks": chunk_count,
                "entities": entity_count # Generic entity count
            }

            # Entity type distribution (counts nodes by their primary label, excluding Document/Chunk)
            entity_types_dist = session.run("""
                MATCH (n) WHERE NOT (n:Document OR n:Chunk) AND size(labels(n)) > 0
                UNWIND labels(n) as label
                RETURN label, count(*) as count
                ORDER BY count DESC LIMIT 20
            """)
            stats["entity_label_distribution"] = [record.data() for record in entity_types_dist]

            # Relationship type distribution (excluding CONTAINS, NEXT)
            rel_types_dist = session.run("""
                MATCH ()-[r]->()
                WHERE NOT type(r) IN ['CONTAINS', 'NEXT', 'MENTIONS'] // Focus on semantic relationships
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC LIMIT 20
            """)
            stats["semantic_relationship_distribution"] = [record.data() for record in rel_types_dist]
            
            return stats
    except Exception as e:
        print(f"❌ Error getting graph statistics: {str(e)}")
        return {"error": str(e)}


def close_connection():
    """Close the Neo4j driver connection"""
    if driver:
        try:
            driver.close()
            print("✅ Neo4j connection closed")
        except Exception as e:
            print(f"❌ Error closing Neo4j connection: {str(e)}")

# Alias for backward compatibility if `create_graph` was used elsewhere
create_graph = create_knowledge_graph


if __name__ == "__main__":
    print("Testing Neo4j Operations (Knowledge Graph Enhanced)...")
    if test_connection():
        print("\n--- Document Stats ---")
        doc_stats = get_document_stats()
        if doc_stats:
            for stat in doc_stats[:3]: # Print first 3
                print(f"  Doc: {stat['document']}, Status: {stat['status']}, Chunks: {stat['chunks_in_graph']}, Entities Mentioned: {stat['unique_entities_mentioned']}")
        else:
            print("  No document stats found.")

        print("\n--- Overall Graph Stats ---")
        graph_stats = get_graph_statistics()
        if graph_stats.get("counts"):
            print(f"  Documents: {graph_stats['counts']['documents']}")
            print(f"  Chunks: {graph_stats['counts']['chunks']}")
            print(f"  Entities: {graph_stats['counts']['entities']}")
            print("  Entity Label Distribution (Top 5):")
            for item in graph_stats.get("entity_label_distribution", [])[:5]:
                print(f"    - {item['label']}: {item['count']}")
            print("  Semantic Relationship Distribution (Top 5):")
            for item in graph_stats.get("semantic_relationship_distribution", [])[:5]:
                 print(f"    - {item['type']}: {item['count']}")
        else:
            print("  No overall graph stats found or error fetching.")
        
        # Example of creating a KG for a dummy document
        # print("\n--- Test KG Creation ---")
        # sample_chunks = [
        #     "Dr. Alice Smith works at Innovatech Inc. located in New York. She is an expert in AI.",
        #     "Innovatech Inc. announced a new product called 'Synergy AI' on March 15, 2024.",
        #     "Synergy AI uses advanced machine learning algorithms. Bob Johnson leads the Synergy AI project."
        # ]
        # try:
        #     create_knowledge_graph(sample_chunks, "dummy_document.txt", domain="technology")
        # except Exception as e:
        #     print(f"Error in dummy KG creation test: {e}")

        # print("\n--- Test Querying ---")
        # query_results = query_knowledge_graph("Alice Smith Innovatech AI", limit=3)
        # print(f"Query results for 'Alice Smith Innovatech AI':")
        # for res in query_results.get("results", []):
        #     print(f"  - Entity: {res['entity_name']} ({res['entity_labels']}), Mentions: {res['entity_mentions']}")
        #     if res.get('direct_connections'):
        #         print("    Connections:")
        #         for conn in res['direct_connections'][:2]: # Show first 2 connections
        #             print(f"      - {conn['relationship_type']} ({conn['direction']}) -> {conn['target_node_name']} ({conn['target_node_labels']})")
        
        # print("\n--- Test Deletion ---")
        # delete_result = delete_document_graph("dummy_document.txt")
        # print(delete_result.get("message", "Deletion status unknown."))

    else:
        print("❌ Neo4j connection test failed. Aborting further tests.")

    close_connection()