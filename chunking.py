# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from PyPDF2 import PdfReader
# import io

# def chunk_pdf(file_bytes):
#     try:
#         reader = PdfReader(io.BytesIO(file_bytes))  # ✅ wrap bytes properly
#         full_text = ""
#         for page in reader.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 full_text += page_text + "\n"

#         if not full_text.strip():
#             raise ValueError("No readable text found in PDF.")

#         splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#         return splitter.split_text(full_text)

#     except Exception as e:
#         raise RuntimeError(f"Failed to process PDF: {str(e)}")


## NEW CODE GOOGLE <><<><><><><><<><><><><><<><>><><><<><><><><><><><<><><<><><><<><><><><><><><><><><><><><

from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader # Corrected import for consistency
import io

def chunk_pdf(file_bytes, filename: str = None): # Added optional filename for context
    try:
        # Consider adding a check for file_bytes type and size early on
        if not file_bytes or not isinstance(file_bytes, bytes):
            raise ValueError("Invalid or empty file_bytes provided.")
        
        reader = PdfReader(io.BytesIO(file_bytes))
        full_text = ""
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
            except Exception as page_e:
                # Log error for specific page but continue if possible
                print(f"⚠️ Warning: Could not extract text from page {i+1} of '{filename or 'unknown file'}'. Error: {page_e}")
                # You might want to add a placeholder like "[UNREADABLE PAGE]" to full_text

        if not full_text.strip():
            # This error should ideally be caught by the caller (upload.py)
            # and returned as a user-friendly message.
            raise ValueError(f"No readable text found in PDF: '{filename or 'unknown file'}'. The PDF might be image-based or password-protected without text layer.")

        # Consistent chunk size with potential LLM context windows
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, # Reasonable default
            chunk_overlap=100, # Good for context continuity
            length_function=len,
            is_separator_regex=False, # Using default separators
        )
        chunks = splitter.split_text(full_text)
        
        if not chunks: # Should be rare if full_text was not empty
            print(f"⚠️ Warning: Text was extracted but splitter returned no chunks for '{filename or 'unknown file'}'. Text length: {len(full_text)}")
            return [] # Return empty list instead of raising another error here

        return chunks

    except ValueError as ve: # Catch specific ValueError from no text
        raise ve # Re-raise to be handled by the caller
    except Exception as e:
        # General catch-all, wrap in RuntimeError for clarity
        print(f"❌ Error during PDF chunking for '{filename or 'unknown file'}': {type(e).__name__} - {e}")
        raise RuntimeError(f"Failed to process PDF '{filename or 'unknown file'}': {str(e)}")

# Added a simple text chunker as discussed for upload.py
def chunk_text(text_content: str, chunk_size: int = 1000, chunk_overlap: int = 100, filename: str = None) -> list[str]:
    try:
        if not text_content or not isinstance(text_content, str):
            raise ValueError("Invalid or empty text_content provided.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = splitter.split_text(text_content)
        
        if not chunks:
            print(f"⚠️ Warning: Text content provided but splitter returned no chunks for '{filename or 'unknown file'}'. Text length: {len(text_content)}")
            return []
            
        return chunks
    except ValueError as ve:
        raise ve
    except Exception as e:
        print(f"❌ Error during text chunking for '{filename or 'unknown file'}': {type(e).__name__} - {e}")
        raise RuntimeError(f"Failed to chunk text content '{filename or 'unknown file'}': {str(e)}")