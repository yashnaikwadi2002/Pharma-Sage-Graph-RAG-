# import hashlib
# import os
# import json

# HASH_FILE = "hashes.json"

# def compute_hash(content: bytes) -> str:
#     return hashlib.sha256(content).hexdigest()

# def hash_exists(hash_val: str) -> bool:
#     if not os.path.exists(HASH_FILE):
#         return False
#     with open(HASH_FILE) as f:
#         hashes = json.load(f)
#     return hash_val in hashes

# def save_hash(hash_val: str, filename: str):
#     if os.path.exists(HASH_FILE):
#         with open(HASH_FILE) as f:
#             hashes = json.load(f)
#     else:
#         hashes = {}
#     hashes[hash_val] = filename
#     with open(HASH_FILE, "w") as f:
#         json.dump(hashes, f)

## NEW CODE GOOGLE ><><><><><><><><<><><><<><><><><<><><<><><><><><><><><><><><><><><><><><><><><><><><<><><><

import hashlib
import os
import json
import threading # For thread-safe access to the hash file
from typing import Optional 

HASH_FILE = os.getenv("HASH_FILE_PATH", "hashes.json") # Allow override via .env
_hash_lock = threading.Lock() # Lock for thread-safe file operations

def compute_hash(content: bytes) -> str:
    if not isinstance(content, bytes):
        raise TypeError("Input content for hashing must be bytes.")
    return hashlib.sha256(content).hexdigest()

def _load_hashes() -> dict:
    """Loads hashes from the JSON file."""
    if not os.path.exists(HASH_FILE):
        return {}
    try:
        with open(HASH_FILE, 'r') as f:
            content = f.read().strip() # Read and strip whitespace
            if not content: # File is empty or only whitespace
                return {}
            return json.loads(content) # Parse JSON from the string content
    except json.JSONDecodeError:
        print(f"⚠️ Warning: Hash file '{HASH_FILE}' is corrupted or not valid JSON. Treating as empty.")
        return {}
    except Exception as e:
        print(f"⚠️ Warning: Could not read hash file '{HASH_FILE}': {e}. Treating as empty.")
        return {}


def _save_hashes(hashes: dict):
    """Saves hashes to the JSON file."""
    try:
        with open(HASH_FILE, 'w') as f:
            json.dump(hashes, f, indent=2) # Added indent for readability
    except Exception as e:
        print(f"❌ Error: Could not write to hash file '{HASH_FILE}': {e}")
        # Consider how to handle this failure; maybe raise an exception

def hash_exists(hash_val: str) -> bool:
    with _hash_lock: # Ensure thread-safe read
        hashes = _load_hashes()
        return hash_val in hashes

def get_filename_for_hash(hash_val: str) -> Optional[str]:
    """Retrieves filename associated with a given hash."""
    with _hash_lock:
        hashes = _load_hashes()
        return hashes.get(hash_val)

def save_hash_entry(hash_val: str, filename: str): # Renamed from save_hash
    """Saves a single hash-filename entry."""
    with _hash_lock: # Ensure thread-safe read-modify-write
        hashes = _load_hashes()
        if hash_val in hashes and hashes[hash_val] != filename:
            print(f"⚠️ Warning: Hash '{hash_val}' already exists with filename '{hashes[hash_val]}'. Overwriting with '{filename}'.")
        hashes[hash_val] = filename
        _save_hashes(hashes)

def remove_hash_entry(hash_val: str) -> bool:
    """Removes a hash entry. Returns True if entry existed and was removed, False otherwise."""
    with _hash_lock:
        hashes = _load_hashes()
        if hash_val in hashes:
            del hashes[hash_val]
            _save_hashes(hashes)
            return True
        return False

def clear_all_hashes():
    """Clears all entries from the hash file."""
    with _hash_lock:
        _save_hashes({}) # Save an empty dictionary
        print(f"ℹ️ All hashes cleared from '{HASH_FILE}'.")

# Example usage (optional)
if __name__ == "__main__":
    print(f"Using hash file: {HASH_FILE}")
    sample_content = b"This is some test content."
    h = compute_hash(sample_content)
    print(f"Computed hash: {h}")

    if not hash_exists(h):
        print("Hash does not exist. Saving...")
        save_hash_entry(h, "test_file.txt")
        print(f"Saved. Hash exists: {hash_exists(h)}")
        print(f"Filename for hash {h}: {get_filename_for_hash(h)}")
    else:
        print(f"Hash {h} already exists for filename: {get_filename_for_hash(h)}.")

    # print("Removing hash...")
    # if remove_hash_entry(h):
    #     print(f"Removed. Hash exists: {hash_exists(h)}")
    # else:
    #     print("Hash not found for removal.")
    
    # print("Clearing all hashes...")
    # clear_all_hashes()
    # print(f"Hashes after clear: {_load_hashes()}")