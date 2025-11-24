Mini Retrieval-Augmented Generation (RAG) Pipeline

This project implements a Mini Retrieval-Augmented Generation (RAG) system that retrieves relevant text documents based on user input using embeddings and similarity search, and optionally generates an answer using a lightweight LLM.

It is a simplified demonstration of how modern systems like ChatGPT Retrieval Mode, Google Search Assistants, Enterprise Knowledge Bots, and AI Search Engines retrieve knowledge before generating responses.

🚀 Features

📂 Loads a subset of the 20 Newsgroups dataset

🔍 Converts documents into vector embeddings

📈 Performs cosine similarity search

🤖 Generates an optional AI answer using a lightweight local LLM

🖥 User-friendly CLI for repeated querying

📊 Dataset Used

This project uses the 20 Newsgroups dataset (from Scikit-Learn).

Selected categories:

sci.space

comp.graphics

rec.sport.baseball

A total of 30 documents are used (10 per category).

🧰 Libraries Used
Library	Purpose
sentence-transformers	Create text embeddings
sklearn.datasets	Load dataset
sklearn.metrics.pairwise	Compute cosine similarity
numpy	Handle indexing & vector operations
gpt4all (optional)	Lightweight local LLM for generation
pandas (optional)	Data formatting/export
