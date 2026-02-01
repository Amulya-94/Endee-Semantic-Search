# Endee Semantic Search Engine

A high-performance semantic search application built using the [Endee Vector Database](https://github.com/EndeeLabs/endee) and `sentence-transformers`. This project demonstrates how vector databases can solve the limitations of traditional keyword search by understanding the *intent* and *context* of user queries.

## 1. Project Overview & Problem Statement

### The Problem
Traditional search engines rely on **keyword matching** (e.g., "Exact Match"). They fail when a user's query uses different words than the document, even if the meaning is the same.
*   *Example*: Searching for "canine" might miss a document containing only "dog".
*   *Limitation*: No understanding of context or semantic relationship.

### The Solution
We built a **Semantic Search Engine** that retrieves documents based on **meaning**. 
By converting text into numerical vectors (embeddings), we can find documents that are mathematically close to the query's meaning, regardless of the specific words used.

## 2. System Design & Technical Approach

The system consists of three main layers:

### A. Ingestion Layer (The "Write" Path)
1.  **Reading**: The system scans Markdown/Text files from the `data/` directory.
2.  **Embedding**: Uses the `sentence-transformers/all-MiniLM-L6-v2` model to convert text chunks into **384-dimensional dense vectors**.
3.  **Indexing**: These vectors are sent to Endee, which indexes them for fast retrieval.

### B. Storage Layer (The Database)
*   **Technology**: Endee Vector Database (running via Docker).
*   **Role**: Stores high-dimensional vectors and performs Approximate Nearest Neighbor (ANN) search using HNSW (Hierarchical Navigable Small World) algorithms for milliseconds-latency retrieval.

### C. Retrieval Layer (The "Read" Path)
1.  **Query Processing**: The user's text query is converted into a vector using the *same* model as ingestion.
2.  **Vector Search**: This query vector is sent to Endee's `/search` API.
3.  **Result Mapping**: Endee returns the IDs of the most similar vectors. The application maps these IDs back to the original text content for display.

## 3. How Endee is Used

Endee is the core **vector store** for this project. We interact with it via its RESTful API:

-   **Index Creation**: We create a specific index (`docs_index`) configured with `cosine` similarity and `384` dimensions.
    *   *Endpoint*: `POST /api/v1/index/create`
-   **Data Insertion**: Generated embeddings are batched and inserted into Endee.
    *   *Endpoint*: `POST /api/v1/index/{name}/vector/insert`
-   **Semantic Querying**: We perform K-Nearest Neighbor (KNN) searches to find the top-5 most relevant results.
    *   *Endpoint*: `POST /api/v1/index/{name}/search`

## 4. Setup and Execution Instructions

### Prerequisites
*   **Docker Desktop** (must be running)
*   **Python 3.8+**
*   **Git**

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd semantic_search_project
```

### Step 2: Start Endee Database
This project requires Endee to be running locally via Docker.
```bash
# Navigate to where you have the Endee docker-compose.yml 
# (Or use the official image directly if you prefer)
docker run -d -p 8080:8080 -v endee-data:/data endeeio/endee-server:latest
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
pip install msgpack
```

### Step 4: Run the Project

**1. Ingest Data**  
Load the sample data (or add your own `.txt` files to `data/`) into the database:
```bash
python main.py ingest
```

**2. Perform a Search**  
Query your data using natural language:
```bash
python main.py search "What features does Endee have?"
```

---
*Assignment submission for Endee.io Placement Drive - Batch 2026*
