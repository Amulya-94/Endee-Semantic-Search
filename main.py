import os
import requests
import sys
import glob
from sentence_transformers import SentenceTransformer

# Configuration
ENDEE_URL = "http://localhost:8080"
INDEX_NAME = "docs_index"
MODEL_NAME = "all-MiniLM-L6-v2"
DIMENSION = 384  # Dimension for all-MiniLM-L6-v2

def check_health():
    try:
        response = requests.get(f"{ENDEE_URL}/api/v1/health")
        if response.status_code == 200:
            print("Endee is reachable.")
            return True
        else:
            print(f"Endee returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("Could not connect to Endee. Is it running?")
        return False

def get_model():
    print(f"Loading model {MODEL_NAME}...")
    return SentenceTransformer(MODEL_NAME)

def create_index():
    print(f"Creating index '{INDEX_NAME}'...")
    payload = {
        "index_name": INDEX_NAME,
        "dim": DIMENSION,
        "space_type": "cosine",  # or l2, ip
        # "precision": "INT8" # Optional, defaults to INT8
    }
    
    # Check if index exists (by trying to list or just create and handle error)
    # Endee's create endpoint might error if it exists.
    response = requests.post(f"{ENDEE_URL}/api/v1/index/create", json=payload)
    
    if response.status_code == 200:
        print("Index created successfully.")
    elif response.status_code == 409:
        print("Index likely already exists.")
    else:
        print(f"Failed to create index: {response.text}")

def ingest_data():
    model = get_model()
    files = glob.glob("data/*.txt") + glob.glob("data/*.md")
    
    if not files:
        print("No files found in data/ directory.")
        return

    vectors = []
    current_id = 0
    
    for file_path in files:
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        for line in lines:
            text = line.strip()
            if not text:
                continue
                
            # Generate embedding
            embedding = model.encode(text).tolist()
            
            # Endee expects: {"id": "...", "vector": [...], "sparse_indices": [], "sparse_values": []}
            # We will use simple ID and dense vector for now.
            vec_obj = {
                "id": str(current_id),
                "vector": embedding,
                # We could add metadata if Endee supported it directly in the vector object like some other DBs,
                # but based on the code, it seems to store vectors.
                # Use external mapping for retrieval if needed, but for now we'll just search.
            }
            vectors.append(vec_obj)
            
            # Map ID to text content (simple in-memory store for this demo)
            # In a real app, this would go into a SQL DB or similar.
            with open("id_map.txt", "a", encoding="utf-8") as mapfile:
                mapfile.write(f"{current_id}|{text}\n")
                
            current_id += 1

    if not vectors:
        print("No data to ingest.")
        return

    # Batch insert
    print(f"Inserting {len(vectors)} vectors...")
    # Endee might have a limit on batch size, let's insert in chunks of 100
    chunk_size = 100
    for i in range(0, len(vectors), chunk_size):
        chunk = vectors[i:i+chunk_size]
        response = requests.post(
            f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/vector/insert",
            json=chunk
        )
        if response.status_code == 200:
            print(f"Inserted chunk {i}-{i+len(chunk)}")
        else:
            print(f"Failed to insert chunk {i}: {response.text}")

def search(query, k=5):
    model = get_model()
    query_vector = model.encode(query).tolist()
    
    payload = {
        "vector": query_vector,
        "k": k,
        "include_vectors": False
    }
    
    response = requests.post(f"{ENDEE_URL}/api/v1/index/{INDEX_NAME}/search", json=payload)
    
    if response.status_code != 200:
        print(f"Search failed: {response.text}")
        return

    # Response is MessagePack. But wait, in main.cpp:
    # resp.add_header("Content-Type", "application/msgpack");
    # So we need to decode it.
    
    try:
        import msgpack
        results = msgpack.unpackb(response.content)
        # results structure depends on SearchResult definition in Endee
        # Usually list of {id, distance, ...}
        
        print(f"\nResults for '{query}':")
        # Load ID map
        id_map = {}
        if os.path.exists("id_map.txt"):
            with open("id_map.txt", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("|", 1)
                    if len(parts) == 2:
                        id_map[parts[0]] = parts[1]

        # The result from Endee seems to be a ResultSet object serialized.
        # Based on typical C++ serialization, it might be a dictionary or list.
        # Let's print raw first to see.
        
        # Assuming typical result format:
        # It references `search_response.value()`
        # Let's assume it returns a list of results.
        
        # We need to handle the msgpack format carefully. 
        # If msgpack isn't installed, we might have issues.
        # But we added it to requirements? No I missed msgpack-python in requirements.
        
        if isinstance(results, dict) and 'results' in results:
             results = results['results'] # Adjust based on actual return structure
        
        # If results is a list of [id, score] tuples or dicts
        for res in results:
            print(f"DEBUG RAW RESULT: {res} (Type: {type(res)})")
            # Inspection of serialization needed to be sure.
            # But let's assume dict access for now.
            doc_id = None
            score = None
            
            # Adjust based on what we see in debugging
            if isinstance(res, dict):
                doc_id = res.get('id') or res.get('label') # VectorObject usually has id
                score = res.get('distance') or res.get('score')
            elif isinstance(res, (list, tuple)):
                 # Observed format: [score, id, ...]
                 score = res[0]
                 doc_id = str(res[1])

            if doc_id:
                doc_id = str(doc_id)
                content = id_map.get(doc_id, "Unknown Content")
                print(f"[ID: {doc_id}] {content[:100]}... (Score: {score})")
            else:
                print(f"Raw result: {res}")

    except Exception as e:
        print(f"Error decoding detailed response: {e}")
        print(f"Raw body size: {len(response.content)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest|search <query>]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "ingest":
        if check_health():
            create_index()
            # Clear id_map
            if os.path.exists("id_map.txt"):
                os.remove("id_map.txt")
            ingest_data()
    elif command == "search":
        if len(sys.argv) < 3:
            print("Please provide a query.")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        if check_health():
            search(query)
    else:
        print("Unknown command.")

if __name__ == "__main__":
    main()
