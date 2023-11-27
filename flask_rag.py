
# This file implments the retrieval part of the RAG system. The following code files comprises 
# of functions and logic to extract the text file, create sentence level embeddings, create vector store 
# using K Nearest Neighbors,Perform chunking and retrieve the relevant chunk from the given query
# 
# Import necessary libraries and modules:
# Flask for creating and managing the web server.
# Request and jsonify from Flask for handling HTTP requests and responses.
from flask import Flask, request, jsonify
# SentenceTransformer for creating sentence embeddings used in NLP applications.
from sentence_transformers import SentenceTransformer
# NearestNeighbors for efficient similarity search in high-dimensional spaces.
from sklearn.neighbors import NearestNeighbors
# Score function from bert_score for evaluating text similarity based on BERT.
from bert_score import score
# Document class from python-docx for reading and manipulating Word documents.
from docx import Document

# Initialize the Flask application to set up the web server.
app = Flask(__name__)

# Define a function to split the input text into smaller chunks based on sentence boundaries.
def chunk_by_sentence(text, max_chunk_length=200):
    # Split the text into a list of sentences using the period as the delimiter.
    paragraphs = text.split(".")
    chunks = []  # Initialize an empty list to store the resulting chunks.

    # Iterate through each sentence/paragraph in the split text.
    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_length:
            # If the paragraph length is within the max_chunk_length, add it as a chunk.
            chunks.append(paragraph)
        else:
            # If the paragraph is too long, split it further into smaller chunks.
            words = paragraph.split()  # Split the paragraph into words.
            # Iterate over the words in steps of max_chunk_length.
            for i in range(0, len(words), max_chunk_length):
                chunk = " ".join(words[i:i+max_chunk_length])  # Join words to form a chunk.
                chunks.append(chunk)  # Add the chunk to the list.
                
    return chunks  # Return the list of chunks.

# Define a similar function to chunk_by_sentence, but this time splitting the text by paragraphs.
def chunk_by_paragraph(text, max_chunk_length=200):
    paragraphs = text.split("\n")  # Split text into paragraphs using newline as the delimiter.
    chunks = []  # Initialize an empty list to store the resulting chunks.

    # Iterate through each paragraph.
    for paragraph in paragraphs:
        if len(paragraph) <= max_chunk_length:
            # Add the paragraph as a chunk if it's within the max length.
            chunks.append(paragraph)
        else:
            # If the paragraph is too long, split it further into smaller chunks.
            words = paragraph.split()  # Split the paragraph into words.
            temp_chunk = ""  # Temporary variable to hold a chunk.
            # Iterate over each word in the paragraph.
            for word in words:
                # Check if adding the word exceeds the max chunk length.
                if len(temp_chunk) + len(word) + 1 <= max_chunk_length:
                    temp_chunk += " " + word  # Add the word to the temp chunk.
                else:
                    # If the chunk reaches max length, add it to the chunks list and reset temp_chunk.
                    chunks.append(temp_chunk.strip())
                    temp_chunk = word
            # Add any remaining text in temp_chunk as the last chunk.
            if temp_chunk:
                chunks.append(temp_chunk.strip())
                
    return chunks  # Return the list of chunks.

# Function to read text from a Word document (.docx file) and return it as a string.
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)  # Open the Word document at the given path.
    full_text = []  # Initialize a list to collect text from each paragraph in the document.
    for para in doc.paragraphs:  # Iterate over each paragraph in the document.
        full_text.append(para.text)  # Append the text of the paragraph to the list.
    return '\n'.join(full_text)  # Join all paragraphs with newline characters and return.

# Extract text from a specific Vastu Shastra Word document and store it in a variable.
vastu_text = extract_text_from_docx('NRM Documentation.docx')

# Function to clean the extracted text by removing non-alphanumeric symbols.
def clean_text(text):
    cleaned_text = ''  # Initialize an empty string to hold the cleaned text.
    # Iterate over each character in the text.
    for char in text:
        # Check if the character is alphanumeric, a space, or a period.
        if char.isalnum() or char.isspace() or char == '.':
            cleaned_text += char  # Add the character to the cleaned text if it meets the criteria.
    return cleaned_text  # Return the cleaned text.

# Clean the extracted Vastu Shastra text using the clean_text function.
vastu_text = clean_text(vastu_text)

# Function to query a Retriever-Generator (RAG) model with a given question.
# It returns the most relevant document chunk based on the question.
def query_rag(question, model, vector_store, document_chunks):
    question_embedding = model.encode([question])  # Generate an embedding for the question.
    _, indices = vector_store.kneighbors(question_embedding)  # Find nearest neighbors for the question embedding.
    # Try to return the most relevant chunk; if an error occurs, return the last chunk.
    try:
        return document_chunks[indices[0][0]]
    except Exception:
        return document_chunks[-1]

# Preprocess the extracted Vastu Shastra text into chunks.
document_chunks = chunk_by_sentence(vastu_text)

# Initialize a SentenceTransformer model for generating embeddings.
model = SentenceTransformer('all-MiniLM-L6-v2')
# Create embeddings for each chunk in the document.
embeddings = model.encode(document_chunks)

# Initialize a NearestNeighbors model to find the nearest neighbor for a given embedding.
vector_store = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
vector_store.fit(embeddings)  # Fit the model on the generated embeddings.

# Function to compare two responses (baseline and RAG-generated) using BERTScore.
def compare_responses(baseline_response, rag_response):
    P, R, F1 = score([rag_response], [baseline_response], lang='en')  # Compute BERTScore.
    return P.mean(), R.mean(), F1.mean()  # Return the mean Precision, Recall, and F1 score.

# Flask route to handle RAG model queries.
# It accepts POST requests and returns an answer generated by the RAG model.
@app.route('/rag', methods=['POST'])
def rag_response():
    data = request.json  # Parse the JSON data sent in the request.
    # Check if the data contains a question; if not, return an error.
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    question = data['question']  # Extract the question from the request data.
    answer_type = data['answer_type']  # Extract the type of answer requested (Long or Short).
    # Choose the appropriate chunking method based on the answer type.
    if answer_type == 'Long':
        document_chunks = chunk_by_paragraph(vastu_text)
    else:
        document_chunks = chunk_by_sentence(vastu_text)
    # Generate an answer using the RAG model.
    answer_rag = query_rag(question, model, vector_store, document_chunks)

    return jsonify({'answer_rag': answer_rag})  # Return the generated answer in JSON format.

# Flask route to compare responses using BERTScore.
# Accepts POST requests and returns BERTScore metrics comparing two responses.
@app.route('/bart_score', methods=['POST'])
def bart_score():
    data = request.json  # Parse the JSON data sent in the request.
    # Check if the necessary data is present; if not, return an error.
    if not data or 'answer_openai' not in data or 'answer_rag' not in data:
        return jsonify({'error': 'No question provided'}), 400
    # Compute BERTScore comparing the two responses.
    P, R, F1 = compare_responses(data['answer_openai'], data['answer_rag'])

    # Return the BERTScore metrics (Precision, Recall, F1 score) in JSON format.
    return jsonify({'Precision': P.numpy().tolist(), 'Recall': R.numpy().tolist(), 'F1 score':F1.numpy().tolist()})

# Main block to run the Flask application if this script is executed directly.
if __name__ == '__main__':
    app.run(debug=True)  # Start the Flask application with debug mode enabled.
