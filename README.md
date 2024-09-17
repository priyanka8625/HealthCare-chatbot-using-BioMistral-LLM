# Medical Chatbot using RAG for Healthcare (Heart Health Focus)

This project aims to build a medical chatbot that can answer questions related to heart health by leveraging modern techniques like Retrieval-Augmented Generation (RAG), fine-tuning, and embedding models. It is designed to deepen the understanding of RAG pipelines and OSS LLMs and implement a complete end-to-end system for querying healthcare documents.

## Table of Contents
- [Overview](#overview)
- [Frameworks Used](#frameworks-used)
- [Models Used](#models-used)
- [Process](#process)
- [How to Run the Project](#how-to-run-the-project)

## Overview

The chatbot is designed to answer queries related to healthcare, specifically heart health, by retrieving relevant information from a provided document [(`HealthyHeart.pdf`)](https://www.nhlbi.nih.gov/files/docs/public/heart/healthyheart.pdf) and generating a response using a large language model (LLM). This involves several steps like document parsing, chunking, embedding generation, and utilizing a vector store to efficiently retrieve relevant text sections.

The end goal of this project is to explore and implement RAG by combining document retrieval with generative responses, enhancing the chatbot's ability to deliver relevant, accurate information efficiently.

## Frameworks Used

### 1. **Langchain** (Pipeline)
   - Provides the necessary infrastructure to create end-to-end chains that combine LLMs with retrievers and other tools. It helps in managing various components, such as document loaders, retrievers, and LLMs.

### 2. **Llama** (LLM Model)
   - An open-source large language model used to handle the natural language generation tasks. Llama is fine-tuned to generate high-quality responses based on the queries and retrieved documents.

### 3. **Sentence-Transformers** (Embedding Model)
   - This framework is used to convert textual data into embedding vectors. It enables the chatbot to understand the meaning and context of the document chunks by generating embeddings that are stored in a vector store.

### 4. **Chroma** (Vector Store)
   - A vector database for efficiently storing and retrieving document embeddings. The chatbot queries this vector store to find relevant sections of the document to retrieve when generating answers.

## Models Used

### 1. **LLM Model: [BioMistral-7b](https://huggingface.co/MaziyarPanahi/BioMistral-7B-GGUF/tree/main)**
   - BioMistral-7B is a large language model specialized for healthcare applications. This model is used to generate the final response based on the query and retrieved documents.

### 2. **Embeddings Model: [PubMedBert-Base](https://huggingface.co/NeuML/pubmedbert-base-embeddings)**
   - PubMedBert-Base embeddings are generated for each document chunk. These embeddings help in comparing the query with document sections to find the most relevant ones.

## Process

The chatbot is built using two key phases:

### 1. **Indexing Phase:**
   - **Step 1: Load the document and parse the text.**
     - The document `HealthyHeart.pdf` is loaded and its content is extracted.
   
   - **Step 2: Divide text into chunks (Chunking).**
     - The extracted text is broken down into smaller, manageable chunks to improve retrieval accuracy.

   - **Step 3: Create embedding vectors for each chunk.**
     - Each chunk is passed through the Sentence-Transformers embedding model (PubMedBert-Base) to generate embedding vectors.

   - **Step 4: Store chunks and embeddings to the Vector Store.**
     - The embeddings and text chunks are stored in the Chroma vector store for efficient retrieval.

### 2. **Querying Phase:**
   - **Step 1: Load the LLM model.**
     - The BioMistral-7B model is loaded to generate responses based on the query and relevant documents.
   
   - **Step 2: Build the application chain end-to-end.**
     - The Langchain framework is used to manage the application chain, combining retrieval and generation processes.

   - **Step 3: Query the chatbot.**
     - A user query is passed to the system, and the following steps occur:
     
        a. The query is passed to the retriever.
        
        b. The retriever searches the vector store using K-Nearest Neighbors (KNN) to find the most relevant document chunks.
        
        c. Both the query and the retrieved chunks are passed to the LLM.
        
        d. The LLM generates a response based on the combined context.

This project uses publically available models and frameworks to demonstrate how RAG pipelines can be leveraged in healthcare. The main focus is on answering questions related to heart health using an open-source document and state-of-the-art models.
