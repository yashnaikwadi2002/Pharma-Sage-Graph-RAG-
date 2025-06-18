
## ASKED TO DELETE BY GOOGLE

# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from backend.core import milvus_ops, neo4j_ops
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# import os
# from dotenv import load_dotenv

# load_dotenv()

# router = APIRouter()

# class QueryRequest(BaseModel):
#     query: str

# class QueryResponse(BaseModel):
#     answer: str
#     sources: list = []

# @router.post("/", response_model=QueryResponse)
# async def query_documents(request: QueryRequest):
#     """
#     Process user query using RAG pipeline:
#     1. Convert query to embedding
#     2. Search similar chunks in Milvus
#     3. Use retrieved context to generate answer
#     """
#     try:
#         # Initialize embeddings model
#         embedder = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=os.getenv("GEMINI_API_KEY")
#         )
        
#         # Convert query to embedding
#         query_embedding = embedder.embed_query(request.query)
        
#         # Search similar chunks in Milvus
#         search_results = milvus_ops.search_similar(query_embedding, top_k=5)
        
#         if not search_results or not search_results[0]:
#             return QueryResponse(
#                 answer="I couldn't find any relevant information in the uploaded documents to answer your question.",
#                 sources=[]
#             )
        
#         # Extract relevant chunks and their metadata
#         relevant_chunks = []
#         sources = []
        
#         for result in search_results[0]:
#             chunk_id = result.entity.get('chunk_id')
#             file_hash = result.entity.get('file_hash')
#             distance = result.distance
            
#             # Get chunk content from Neo4j (optional - you could also store content in Milvus)
#             chunk_content = get_chunk_content_from_neo4j(chunk_id)
#             if chunk_content:
#                 relevant_chunks.append(chunk_content)
#                 sources.append({
#                     "chunk_id": chunk_id,
#                     "file_hash": file_hash,
#                     "relevance_score": float(distance)
#                 })
        
#         if not relevant_chunks:
#             return QueryResponse(
#                 answer="I found some potentially relevant documents, but couldn't retrieve their content.",
#                 sources=sources
#             )
        
#         # Generate answer using retrieved context
#         context = "\n\n".join(relevant_chunks)
#         answer = generate_answer_with_context(request.query, context)
        
#         return QueryResponse(answer=answer, sources=sources)
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

# def get_chunk_content_from_neo4j(chunk_id: str) -> str:
#     """
#     Retrieve chunk content from Neo4j database
#     """
#     try:
#         with neo4j_ops.driver.session() as session:
#             result = session.run(
#                 "MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c.content as content",
#                 chunk_id=chunk_id
#             )
#             record = result.single()
#             return record["content"] if record else ""
#     except Exception as e:
#         print(f"Error retrieving chunk content: {e}")
#         return ""

# def generate_answer_with_context(query: str, context: str) -> str:
#     """
#     Generate answer using Gemini with retrieved context
#     """
#     try:
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             google_api_key=os.getenv("GEMINI_API_KEY"),
#             temperature=0.3
#         )
        
#         prompt = f"""
#         You are a pharmaceutical knowledge assistant. Use the following context from pharmaceutical documents to answer the user's question accurately.
        
#         Context from documents:
#         {context}
        
#         Question: {query}
        
#         Instructions:
#         - Answer based primarily on the provided context
#         - If the context doesn't contain enough information, say so clearly
#         - Be precise and cite specific information when possible
#         - Focus on pharmaceutical and medical accuracy
#         - If asked about dosages, side effects, or medical advice, remind users to consult healthcare professionals
        
#         Answer:
#         """
        
#         response = llm.invoke(prompt)
#         return response.content
        
#     except Exception as e:
#         return f"Error generating answer: {str(e)}"

# @router.delete("/clear")
# async def clear_all_documents():
#     """
#     Clear all documents from both Milvus and Neo4j
#     """
#     try:
#         # Clear Milvus collection
#         milvus_ops.drop_collection()
        
#         # Clear Neo4j database (all documents and chunks)
#         with neo4j_ops.driver.session() as session:
#             session.run("MATCH (n) DETACH DELETE n")
        
#         # Clear hash file
#         import os
#         if os.path.exists("hashes.json"):
#             os.remove("hashes.json")
        
#         return {"message": "All documents cleared successfully"}
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")