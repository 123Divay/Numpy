# mini_rag_pipeline_with_comments.py

# Import dataset loader from scikit-learn
from sklearn.datasets import fetch_20newsgroups

# Import SentenceTransformer for embeddings (vector representation of text)
from sentence_transformers import SentenceTransformer

# Import cosine similarity to compare embeddings
from sklearn.metrics.pairwise import cosine_similarity

# Import numpy for indexing and sorting
import numpy as np
# from openai import OpenAI


# ---------------------------------------------------------
# 1️⃣ Load dataset: Fetch 20 Newsgroup documents
# ---------------------------------------------------------

# Define the categories we want to load (3 categories from 20 Newsgroups)
categories = ['sci.space', 'comp.graphics', 'rec.sport.baseball']

# Download only the training dataset for the selected categories
# remove=('headers', 'footers', 'quotes') cleans the text content
dataset = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# We will store selected document texts and labels here
docs = []
labels = []

# Loop through each category separately to collect 10 docs from each
for cat in categories:
    count = 0  # how many docs we have collected for this category

    # zip(dataset.data, dataset.target) returns text + numeric label pairs
    for text, label in zip(dataset.data, dataset.target):

        # dataset.target_names[label] converts numeric label → category name
        if dataset.target_names[label] == cat:
            docs.append(text)     # store document text
            labels.append(cat)    # store its category name
            count += 1            # increment counter

        # Stop after collecting exactly 10 docs for the category
        if count >= 10:
            break

# Show how many docs were loaded (should be 30)
print(f"Loaded {len(docs)} documents in total.")
print(f"Categories: {set(labels)}\n")  # Print unique category names collected


# ---------------------------------------------------------
# 2️⃣ Create embeddings for each document
# ---------------------------------------------------------

# Load a pre-trained embedding model (384-dim vectors)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert all document texts into embeddings (vector form)
# show_progress_bar=True shows progress while embedding
doc_embeddings = model.encode(docs, show_progress_bar=True)


# ---------------------------------------------------------
# 3️⃣ Function: Retrieve most similar documents to a query
# ---------------------------------------------------------

def retrieve_top_docs(query, top_k):
    """
    Given a user query, return the top_k most similar documents
    using cosine similarity on embeddings.
    """

    # Convert query text into an embedding vector
    query_embedding = model.encode([query])

    # Compare query embedding vs document embeddings
    # cosine_similarity returns an array of similarity scores
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    

    # argsort() returns indices sorted by value (ascending)
    # [::-1] reverses to descending → highest similarity first
    # [:top_k] picks top K highest-ranked indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Build and return list of (doc_text, doc_label, similarity_score)
    return [(docs[i], labels[i], similarities[i]) for i in top_indices]


# ---------------------------------------------------------
# 4️⃣ Run Retrieval: Ask a question
# ---------------------------------------------------------

i = 0  # initialize i before using it

while i >= 0:
    var = input("Do you want to provide a query? Press Y to continue: ")

    if var == "y" or var == "Y":
        askquery()   # your function that handles the query
    else:
        print("Exiting...")
        break        # stop the loop properly


def askquery():
    query = input("\n🔍 Enter your query: ")
    top_k = int(input("📄 How many top documents do you want to retrieve? (e.g., 3): "))
    top_docs = retrieve_top_docs(query,top_k)

# Print what was retrieved
    print(f"🔍 Query: {query}\n")
    print("Top Retrieved Documents:")

    for i, (doc, label, score) in enumerate(top_docs, 1):

    # Show only the first 200 characters of each document
        snippet = doc[:200].replace('\n', ' ')

        print(f"{i}. [{label}] (score={score:.3f}) → {snippet}...\n")


# ---------------------------------------------------------
# 5️⃣ (Optional) Generate a final answer from retrieved docs
# ---------------------------------------------------------

# Combine retrieved documents into a single context string
#     context = " ".join([d for d, _, _ in top_docs])

#     client = OpenAI(api_key="YOUR_API_KEY")

# # 'context' is the retrieved text from your top_docs
#     response = client.chat.completions.create(
#         model="gpt-5",
#         messages=[
#             {"role": "system", "content": "Use the context to answer the question. Do not hallucinate."},
#             {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
#         ]
#     )

#     print("LLM Final Answer:", response.choices[0].message["content"])
    return

