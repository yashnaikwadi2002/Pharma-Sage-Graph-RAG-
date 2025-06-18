# import streamlit as st
# import requests
# import json

# # Configure page
# st.set_page_config(page_title="Pharma RAG", layout="wide")
# st.title("🧪 PHARMA SAGE")

# # API base URL
# API_BASE = "http://localhost:8000/api"

# # Initialize session state for chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Define columns layout
# col1, col2 = st.columns([2, 1])

# # === Chat Interface ===
# with col1:
#     st.header("💬 Pharma Knowledge Assistant")
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#             if message["role"] == "assistant" and "sources" in message:
#                 with st.expander("📚 Sources"):
#                     for source in message["sources"]:
#                         st.write(f"• Chunk ID: {source['chunk_id']}")
#                         st.write(f"  Relevance: {source['similarity']:.3f}")

#     if prompt := st.chat_input("Ask about pharmaceutical documents..."):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             with st.spinner("Thinking with Gemini..."):
#                 try:
#                     response = requests.post(
#                         f"{API_BASE}/intelligent/",
#                         json={"query": prompt, "mode": "hybrid"},
#                         timeout=60
#                     )
#                     if response.status_code == 200:
#                         data = response.json()
#                         st.markdown(data.get("contextual_answer", "No answer generated."))
#                         st.session_state.messages.append({
#                             "role": "assistant",
#                             "content": data.get("contextual_answer", "No answer."),
#                             "sources": data.get("semantic_results", [])
#                         })
#                     else:
#                         st.error(f"Error: {response.status_code}")
#                 except Exception as e:
#                     st.error(f"Request failed: {e}")

# # === Upload Panel ===
# with col2:
#     st.header("📂 Upload PDFs")
#     uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
#     if uploaded_files:
#         for file in uploaded_files:
#             with st.spinner(f"Uploading {file.name}..."):
#                 try:
#                     files = {"file": (file.name, file.getvalue(), "application/pdf")}
#                     response = requests.post(f"{API_BASE}/upload/", files=files)
#                     result = response.json()
#                     status = result.get("status", "Unknown")
#                     if status == "Success":
#                         st.success(f"✅ {file.name} uploaded successfully")
#                     elif status == "Duplicate":
#                         st.info(f"ℹ️ {file.name} already exists")
#                     else:
#                         st.error(f"❌ {file.name}: {result.get('error', 'Upload failed')}")
#                 except Exception as e:
#                     st.error(f"❌ {file.name} upload error: {str(e)}")

#     st.divider()

#     # Document Management Section
#     st.subheader("📋 Document Management")
    
#     # Get list of documents for the search functionality
#     try:
#         docs_response = requests.get(f"{API_BASE}/documents/", timeout=10)
#         if docs_response.status_code == 200:
#             documents = docs_response.json()
#             doc_names = [doc['name'] for doc in documents if isinstance(doc, dict) and 'name' in doc]
#         else:
#             doc_names = []
#             st.error("Failed to fetch document list")
#     except Exception as e:
#         doc_names = []
#         st.error(f"Error fetching documents: {str(e)}")

#     # Search and Delete Section
#     if doc_names:
#         st.write("**Delete Specific Document:**")
        
#         # Search bar with selectbox for better UX
#         selected_doc = st.selectbox(
#             "Select document to delete:",
#             options=[""] + doc_names,
#             index=0,
#             help="Choose a document from the dropdown to delete"
#         )
        
#         # Alternative: Text input with filtering (uncomment if you prefer search bar)
#         search_term = st.text_input("🔍 Search document name:", placeholder="Type to search...")
#         if search_term:
#             filtered_docs = [doc for doc in doc_names if search_term.lower() in doc.lower()]
#             if filtered_docs:
#                 selected_doc = st.selectbox("Select from filtered results:", [""] + filtered_docs)
#             else:
#                 st.warning("No documents match your search")
#                 selected_doc = ""
        
#         # Delete button
#         if selected_doc:
#             col_del1, col_del2 = st.columns([1, 1])
            
#             with col_del1:
#                 if st.button(f"🗑️ Delete", key="delete_specific", use_container_width=True):
#                     # Confirmation step
#                     if not st.session_state.get(f"confirm_delete_{selected_doc}", False):
#                         st.session_state[f"confirm_delete_{selected_doc}"] = True
#                         st.warning(f"⚠️ Click again to confirm deletion of '{selected_doc}'")
#                         st.rerun()
#                     else:
#                         # Perform deletion
#                         with st.spinner(f"Deleting {selected_doc}..."):
#                             try:
#                                 delete_response = requests.delete(
#                                     f"{API_BASE}/documents/{selected_doc}",
#                                     timeout=30
#                                 )
                                
#                                 if delete_response.status_code == 200:
#                                     result = delete_response.json()
#                                     st.success(f"✅ {selected_doc} deleted successfully!")
#                                     # Clear confirmation state
#                                     if f"confirm_delete_{selected_doc}" in st.session_state:
#                                         del st.session_state[f"confirm_delete_{selected_doc}"]
#                                     st.rerun()
#                                 else:
#                                     error_msg = delete_response.json().get('error', 'Unknown error')
#                                     st.error(f"❌ Failed to delete: {error_msg}")
#                                     # Clear confirmation state on error
#                                     if f"confirm_delete_{selected_doc}" in st.session_state:
#                                         del st.session_state[f"confirm_delete_{selected_doc}"]
                                        
#                             except Exception as e:
#                                 st.error(f"❌ Delete error: {str(e)}")
#                                 # Clear confirmation state on error
#                                 if f"confirm_delete_{selected_doc}" in st.session_state:
#                                     del st.session_state[f"confirm_delete_{selected_doc}"]
            
#             with col_del2:
#                 if st.button("❌ Cancel", key="cancel_delete", use_container_width=True):
#                     # Clear any confirmation states
#                     keys_to_remove = [key for key in st.session_state.keys() if key.startswith("confirm_delete_")]
#                     for key in keys_to_remove:
#                         del st.session_state[key]
#                     st.rerun()
#     else:
#         st.info("No documents available to delete")

#     st.divider()

#     # Existing buttons
#     col_a, col_b, col_c = st.columns(3)
#     with col_a:
#         if st.button("📄 List Documents"):
#             try:
#                 docs = requests.get(f"{API_BASE}/documents/").json()
#                 st.subheader("📋 Uploaded Documents")
#                 for doc in docs:
#                     st.write(f"• {doc['name']} ({doc['status']})")
#             except Exception as e:
#                 st.error(f"List error: {e}")

#     with col_b:
#         if st.button("🗑️ Clear All"):
#             try:
#                 response = requests.delete(f"{API_BASE}/query/clear")
#                 if response.status_code == 200:
#                     st.success("All data cleared")
#                     st.session_state.messages = []
#                     st.rerun()
#                 else:
#                     st.error("Failed to clear")
#             except Exception as e:
#                 st.error(f"Clear error: {e}")

# # === Sidebar ===
# with st.sidebar:
#     st.header("🔧 Status")
#     try:
#         # Use the existing API_BASE - don't add /api again
#         response = requests.get(f"{API_BASE}/intelligent/health/", timeout=5)
#         if response.status_code == 200:
#             health = response.json()
#             if "components" in health:
#                 for key, value in health["components"].items():
#                     st.write(f"{key.title()}: {'✅' if value == 'healthy' else '❌'}")
#             else:
#                 st.write("✅ Backend is healthy")
#         else:
#             st.error(f"❌ Backend returned status: {response.status_code}")
#     except Exception as e:
#         st.error(f"❌ Backend error: {str(e)}")

#     st.divider()
#     st.header("ℹ️ Help")
#     st.markdown("""
#     - Upload pharmaceutical PDFs
#     - Ask domain-specific questions
#     - Review AI-backed graph answers
#     """)
#     if st.button("🧹 Clear Chat"):
#         st.session_state.messages = []
#         st.rerun()


# NEW CODE GOOGLE >>><><<><><><><><><><<>><><><><><><><><><><><<><><><><><><><<><><><><><><><><><><><><>

# frontend/app.py

import streamlit as st
import requests
import json
import urllib.parse # For URL encoding filenames

# --- Page Configuration ---
st.set_page_config(page_title="Pharma Sage KG", layout="wide", initial_sidebar_state="expanded")
st.title("🧪 PHARMA SAGE - Knowledge Graph Edition")

# --- API Configuration ---
API_BASE_URL = "http://localhost:8000/api" # Ensure this matches your backend setup

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_list" not in st.session_state:
    st.session_state.document_list = []
if "last_doc_fetch_error" not in st.session_state:
    st.session_state.last_doc_fetch_error = None

# --- Helper Functions ---
def fetch_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents/", timeout=60)
        response.raise_for_status() # Raise an exception for HTTP error codes
        st.session_state.document_list = response.json()
        st.session_state.last_doc_fetch_error = None
    except requests.exceptions.RequestException as e:
        st.session_state.document_list = []
        st.session_state.last_doc_fetch_error = f"Failed to fetch document list: {e}"
        print(f"Error fetching documents: {e}") # Log to console
    except json.JSONDecodeError as e:
        st.session_state.document_list = []
        st.session_state.last_doc_fetch_error = f"Failed to parse document list: {e}"
        print(f"Error parsing document list JSON: {e}")

# Fetch documents on initial load or if list is empty
if not st.session_state.document_list:
    fetch_documents()

# --- UI Layout ---
col_chat, col_sidebar_content = st.columns([2, 1]) # Main chat area and right panel

# === Chat Interface (Left Column) ===
with col_chat:
    st.header("💬 Pharma Knowledge Assistant")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("details"):
                with st.expander("🔍 Assistant's Search Details", expanded=False):
                    details = message["details"]
                    if details.get("semantic_search_results"):
                        st.subheader("📌 Semantic Matches (Chunks):")
                        for hit in details["semantic_search_results"][:3]: # Show top 3
                            st.markdown(f"""
                                - **Doc:** `{hit.get('document_name', 'N/A')}` Chunk {hit.get('chunk_index', 'N/A')}
                                - **Preview:** *"{hit.get('content_preview', 'N/A')}"*
                                - **Similarity:** {hit.get('similarity', 0.0):.3f}
                            """)
                    if details.get("graph_query_results"):
                        st.subheader("🔗 Graph Matches (Entities):")
                        for hit in details["graph_query_results"][:3]: # Show top 3
                             st.markdown(f"""
                                - **Entity:** `{hit.get('entity_name', 'N/A')}` (Type: {', '.join(hit.get('entity_labels', []))})
                                - **Mentions:** {hit.get('entity_mentions', 'N/A')}
                                - **Description:** *{hit.get('entity_description', 'N/A')}*
                            """)
                    if details.get("graph_entity_context_chunks"):
                        st.subheader("📄 Context from Graph Entities (Chunks):")
                        for hit in details["graph_entity_context_chunks"][:3]: # Show top 3
                            st.markdown(f"""
                                - **Doc:** `{hit.get('document_name', 'N/A')}` Chunk {hit.get('chunk_index', 'N/A')} (related to entity: `{hit.get('matched_entity')}`)
                                - **Preview:** *"{hit.get('content_preview', 'N/A')}"*
                            """)


    # Chat input
    if user_query := st.chat_input("Ask about pharmaceutical documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            with st.spinner("🧠 Pharma Sage is thinking... (This may take a moment for complex queries)"):
                try:
                    payload = {
                        "query": user_query,
                        "mode": "hybrid", # Always use hybrid for the best results
                        "limit": 5, # Request a few more results for better context
                        "include_contextual_summary": True
                    }
                    response = requests.post(f"{API_BASE_URL}/intelligent/", json=payload, timeout=90) # Increased timeout
                    response.raise_for_status()
                    
                    data = response.json()
                    answer = data.get("contextual_summary", "I received a response, but no summary was generated.")
                    if not answer and (data.get("semantic_search_results") or data.get("graph_query_results")):
                        answer = "I found some information but couldn't form a direct summary. See details below."
                    elif not answer:
                        answer = "I couldn't find a specific answer or relevant information for your query."

                    full_response_content = answer
                    message_placeholder.markdown(full_response_content)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response_content,
                        "details": { # Store all relevant parts for the expander
                            "semantic_search_results": data.get("semantic_search_results"),
                            "graph_query_results": data.get("graph_query_results"),
                            "graph_entity_context_chunks": data.get("graph_entity_context_chunks")
                        }
                    })

                except requests.exceptions.RequestException as e:
                    error_msg = f"Network or API Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except json.JSONDecodeError as e:
                    error_msg = f"API Error: Could not parse response. {response.text if 'response' in locals() else ''}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"An unexpected error occurred: {e}"
                    st.error(error_msg)
                    traceback.print_exc()
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# === Right Panel (Upload & Document Management) ===
with col_sidebar_content:
    st.header("📦 Document Pipeline")
    
    # --- Upload Section ---
    with st.expander("📤 Upload New PDF Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Select PDF files to process into the Knowledge Graph:",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        files_payload = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_BASE_URL}/upload/", files=files_payload, timeout=1600) # Increased timeout
                        response.raise_for_status()
                        result = response.json()
                        
                        if result.get("status") == "Success":
                            st.success(f"✅ '{uploaded_file.name}' processed: {result.get('chunks_created', 'N/A')} chunks. Domain: {result.get('domain_used_for_kg', 'N/A')}.")
                            fetch_documents() # Refresh document list
                            st.rerun() # Rerun to update UI elements dependent on document_list
                        elif result.get("status") == "Duplicate Content":
                            st.info(f"ℹ️ '{uploaded_file.name}' (or its content) has already been processed.")
                        else:
                            st.error(f"❌ '{uploaded_file.name}': {result.get('detail', result.get('error', 'Upload failed'))}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"❌ Network error uploading '{uploaded_file.name}': {e}")
                    except Exception as e:
                        st.error(f"❌ Error processing '{uploaded_file.name}': {e}")
            # Clear uploader after processing to prevent re-upload on rerun
            # This is a known Streamlit trick; might need adjustment based on exact behavior
            # st.session_state.pdf_uploader = [] # This can sometimes cause issues, test it.


    st.divider()

    # --- Document Management Section ---
    st.subheader("📚 Manage Processed Documents")

    if st.button("🔄 Refresh Document List", key="refresh_docs"):
        fetch_documents()
        st.rerun()

    if st.session_state.last_doc_fetch_error:
        st.error(st.session_state.last_doc_fetch_error)

    if st.session_state.document_list:
        for doc_data in st.session_state.document_list:
            doc_name = doc_data.get("name", "Unknown Document")
            with st.expander(f"📄 {doc_name} (Status: {doc_data.get('status', 'N/A')})"):
                st.markdown(f"**Total Chunks:** {doc_data.get('total_chunks_in_doc', 'N/A')}")
                st.markdown(f"**Chunks Processed for KG:** {doc_data.get('chunks_processed_for_kg', 'N/A')}")
                st.markdown(f"**Unique Entities Mentioned:** {doc_data.get('unique_entities_mentioned', 'N/A')}")
                st.markdown(f"**Created At:** {doc_data.get('created_at', 'N/A')}")
                st.markdown(f"**KG Processing Completed At:** {doc_data.get('kg_processing_completed_at', 'N/A')}")
                if doc_data.get('error_message'):
                    st.error(f"**Error:** {doc_data.get('error_message')}")

                if st.button(f"🗑️ Delete '{doc_name}'", key=f"delete_{doc_name}"):
                    # Confirmation for deletion
                    if st.session_state.get(f"confirm_delete_{doc_name}", False):
                        with st.spinner(f"Deleting '{doc_name}'..."):
                            try:
                                # URL encode the filename for the path parameter
                                encoded_filename = urllib.parse.quote(doc_name)
                                delete_response = requests.delete(f"{API_BASE_URL}/documents/{encoded_filename}", timeout=60)
                                delete_response.raise_for_status()
                                result = delete_response.json()
                                st.success(f"✅ '{doc_name}' deletion initiated. Details: {result.get('message')}")
                                log_details = result.get('details', {})
                                for k, v in log_details.items():
                                    st.info(f"  - {k.replace('_', ' ').title()}: {v}")
                                fetch_documents() # Refresh list
                                del st.session_state[f"confirm_delete_{doc_name}"] # Clear confirmation
                                st.rerun()
                            except requests.exceptions.RequestException as e:
                                st.error(f"❌ Failed to delete '{doc_name}': {e}")
                            except Exception as e:
                                st.error(f"❌ Unexpected error deleting '{doc_name}': {e}")
                        
                    else:
                        st.session_state[f"confirm_delete_{doc_name}"] = True
                        st.warning(f"Are you sure you want to delete '{doc_name}'? Click delete again to confirm.")
                        st.rerun() # Rerun to show confirmation state
    else:
        st.info("No documents processed yet, or unable to fetch document list.")


# === Sidebar (Status & Admin Actions) ===
with st.sidebar:
    st.header("⚙️ System Status & Admin")
    
    # --- Health Check ---
    if st.button("🩺 Check System Health", key="health_check_btn"):
        with st.spinner("Checking health..."):
            try:
                response = requests.get(f"{API_BASE_URL}/intelligent/health/", timeout=60)
                response.raise_for_status()
                health_data = response.json()
                
                st.subheader(f"Overall Status: {health_data.get('overall_status', 'Unknown').upper()}")
                
                components = health_data.get("components", {})
                if components:
                    st.markdown("**Components:**")
                    for comp_name, comp_status in components.items():
                        status_icon = "✅" if comp_status.get("status") == "healthy" else "❌"
                        details = comp_status.get("details") or comp_status.get("message") or comp_status.get("error") or ""
                        st.markdown(f"- {comp_name.replace('_', ' ').title()}: {status_icon} {comp_status.get('status','N/A')} {f'({details})' if details else ''}")
                
                kg_overview = health_data.get("knowledge_graph_overview", {})
                if kg_overview and not kg_overview.get("error"):
                    st.markdown("**Knowledge Graph Overview:**")
                    st.markdown(f"- Documents: {kg_overview.get('documents', 'N/A')}")
                    st.markdown(f"- Chunks: {kg_overview.get('chunks', 'N/A')}")
                    st.markdown(f"- Entities: {kg_overview.get('entities', 'N/A')}")
                elif kg_overview.get("error"):
                     st.warning(f"KG Overview Error: {kg_overview.get('error')}")

            except requests.exceptions.RequestException as e:
                st.error(f"❌ Health check failed: {e}")
            except Exception as e:
                st.error(f"❌ Error processing health data: {e}")
    
    st.divider()

    # --- Admin Actions ---
    st.subheader("⚠️ Admin Actions")
    if st.button("🧹 Clear Chat History", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("💣 Wipe Entire Knowledge Base & Data", key="wipe_all_data", help="This will delete all uploaded documents, KG data, and vector embeddings. This action is IRREVERSIBLE."):
        if st.session_state.get("confirm_wipe_all", False):
            with st.spinner("🚨 Wiping all data... This is irreversible!"):
                try:
                    # This endpoint is in documents.py, prefixed with /api/documents
                    response = requests.delete(f"{API_BASE_URL}/documents/admin/clear-all-data", timeout=120)
                    response.raise_for_status()
                    result = response.json()
                    st.success(f"✅ All data wipe initiated. {result.get('message')}")
                    for log_item in result.get('summary_log', result.get('deletion_log', [])): # Handle key variations
                        st.info(log_item)
                    fetch_documents() # Refresh lists
                    st.session_state.messages = [] # Clear chat
                    del st.session_state["confirm_wipe_all"]
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ Failed to wipe data: {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error during data wipe: {e}")
        else:
            st.session_state["confirm_wipe_all"] = True
            st.warning("DANGER ZONE! Are you absolutely sure you want to delete ALL data? Click the button again to confirm.")
            st.rerun()


    st.divider()
    st.markdown("--- \n *Pharma Sage v1.2.0*")