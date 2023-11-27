# This files integrates the output from flask_rag.py, passes the query and the retrieved document 
#chunk to the llm to generate a response. The file also uses call to OpenAI via api to generate 
#the base response. The responses, along with BERT score is then displayed on streamlit

# Import necessary libraries for the application:
# Altair for data visualization.
import altair as alt
# Streamlit for building and running the web application.
import streamlit as st
# Import OpenAI, PromptTemplate, and LLMChain from langchain for handling language models.
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
# Requests library for making HTTP requests.
import requests
# Document class from python-docx for reading Word documents.
from docx import Document
# Import json for parsing JSON files.
import json
# Import os for accessing environment variables and file system.
import os

# Import the ChatOpenAI model from langchain.
from langchain.chat_models import ChatOpenAI

# Configure the Streamlit app layout to be wide, providing more space for elements.
st.set_page_config(layout="wide")

# Set the rendering options for Altair visualizations, scaling them for better visibility.
alt.renderers.set_embed_options(scaleFactor=2)

# Load API keys from a JSON file for authentication purposes.
with open('api_keys.json') as api_file:
    api_dict = json.load(api_file)
# Set the OpenAI API key in the environment variable for use in the application.
os.environ['OPENAI_API_KEY'] = api_dict['API OPENAI']

# Define a template for long answer responses. This template structures how the responses should be formed.
template_for_long_answer = """
Answer question from NRM document, The answer should be from the provided document only

Question:
    {question}

Instructions:
    1. Be very specific to the document I have in conversation history
    2. Do not refer to anything else external from the source

Answer:
"""

# Initialize the GPT model and store it in the Streamlit session state for persistent access across reruns.
st.session_state['GPT'] = OpenAI()  # Alternatively, you can use a specific model like ChatOpenAI with desired settings.

# Function to extract text from a Word document (.docx file).- Vaastu Shastra
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)  # Open the Word document.
    full_text = []  # Initialize a list to hold the text of each paragraph.
    for para in doc.paragraphs:  # Iterate over each paragraph in the document.
        full_text.append(para.text)  # Append the paragraph text to the list.
    return '\n'.join(full_text)  # Join all paragraphs with newline and return the combined text.

# Extract and store the text from a Vastu Shastra document for use in the application.
vastu_text = extract_text_from_docx('NRM Documentation.docx')

# Function to clean the extracted text by removing unwanted characters for optimized API calls.
def clean_text(text):
    cleaned_text = ''  # Initialize an empty string for the cleaned text.
    for char in text:
        # Include only alphanumeric characters, spaces, and periods in the cleaned text.
        if char.isalnum() or char.isspace() or char == '.':
            cleaned_text += char
    return cleaned_text  # Return the cleaned text.

# Clean the Vastu Shastra text using the clean_text function.
vastu_text = clean_text(vastu_text)

# Check if the 'responses' key exists in the session state; if not, initialize it as an empty dictionary.
if not 'responses' in st.session_state:
    st.session_state['responses'] = {}

# Main function of the Streamlit app, containing the app's logic and UI elements.
def main():
    # Iterate over stored responses and display them in the app.
    for question_i, answer_i in st.session_state['responses'].items():
        # Create an expander for each question and response pair for better UI organization.
        col11, col12 = st.expander(question_i).columns([1, 1])
        col11.markdown('### Response from OpenAI API')
        col12.markdown('### Response from RAG')
        col11.markdown(answer_i[0])  # Display the response from OpenAI API.
        col12.markdown(answer_i[1])  # Display the response from RAG model.

    # UI elements for user input: question field, model settings, and an 'Ask' button.
    question_input = st.text_input("Ask your question: ")
    setting_cols = st.columns([1, 1, 1])
    temperature = setting_cols[0].number_input("Choose temperature of the model: ")
    answer_type = setting_cols[1].selectbox('Choose Answer type: ', ["Long", "Short"])
    answer_button = st.button('Ask to PricingGPT')

    # Logic to handle when the 'Ask' button is pressed.
    if answer_button:
        llm = st.session_state['GPT']  # Retrieve the GPT model from the session state.
        # Define a template for baseline responses.
        template_for_baseline = """
            Answer question from Vastu Shastra document:

            Question:
                {question}

            Instructions:
                1. Be very specific to the document I have in conversation history
                2. Do not refer to anything else external from the source
                3. Answer should be {answer_type} answer type

            Answer:
            """
        # Create a prompt template object for generating prompts based on the template.
        prompt_repo = PromptTemplate(
            template=template_for_baseline,
            input_variables=['question', 'answer_type'],
        )
        # Define a specific question for testing purposes (this seems to be a leftover from testing).
        question = "What does Sales Analytics do?"

        # Create an LLMChain object for running the language model with the defined prompt.
        llm_chain = LLMChain(prompt=prompt_repo, llm=llm)
        # Execute the language model chain to generate an answer.
        answer = llm_chain.run(
            conversation_history=vastu_text,
            question=question_input,
            answer_type=answer_type
        )

        # Prepare and make a POST request to the Flask server to get a response from the RAG model.
        url = 'http://127.0.0.1:5000/'
        rag_endpoint = 'rag'
        score_endpoint = 'bart_score'
        data = {'question': question_input, 'answer_type': answer_type}
        answer_rag = requests.post(url + rag_endpoint, json=data).json()['answer_rag']

        # Define a template for processing the RAG response with the language model.
        template_for_rag_llm = """
            Answer question from Vastu Shastra document:

            Question:
                {question}

            Instructions:
                1. Be very specific to the text I have in conversation history
                2. Do not refer to anything else external from the source
                3. Answer should be {answer_type} answer type

            Answer:
            """
        # Create another prompt template for the RAG response.
        prompt_repo = PromptTemplate(
            template=template_for_rag_llm,
            input_variables=['question', 'answer_type']
        )

        # Execute the language model chain with the RAG response as the conversation history.
        llm_chain = LLMChain(prompt=prompt_repo, llm=llm)
        answer_rag_llm = llm_chain.run(
            conversation_history=answer_rag,
            question=question_input,
            answer_type=answer_type
        )

        # Store the generated responses in the session state.
        st.session_state['responses'][question_input] = [answer, answer_rag_llm]
        # Create columns for displaying the responses in the UI.
        col1, col2 = st.columns([1, 1])

        col1.markdown('### Response from OpenAI API')
        col2.markdown('### Response from RAG')

        col1.markdown(answer)  # Display the OpenAI API response.
        col2.markdown(answer_rag_llm)  # Display the RAG response.

        # Prepare and send a request to compare the responses using BERTScore.
        data = {'answer_rag': answer_rag_llm, 'answer_openai': answer}
        bart_score = requests.post(url + score_endpoint, json=data).json()

        # Display the comparison scores in the UI.
        st.markdown('#### Comparison Score')
        st.write(bart_score)

        # Define a template for calculating the coherence score between the two responses.
        template_for_coherence_score = """
            Answer in single or two words only.

            rag_llm_response: {answer_rag_llm}

            llm_response: {answer}

            Answer:
            <Coherence level>
            """
        # Create a prompt template for the coherence score calculation.
        prompt_repo = PromptTemplate(
            template=template_for_coherence_score,
            input_variables=['answer_rag_llm', 'answer']
        )

        # Execute the language model chain to calculate the coherence score.
        llm_chain = LLMChain(prompt=prompt_repo, llm=llm)
        answer_rag_llm_coherence_score = llm_chain.run(
            answer_rag_llm=answer_rag_llm,
            answer=answer
        )

        # Display the coherence score in the UI.
        st.markdown('#### Coherence Score')
        st.write(answer_rag_llm_coherence_score)

# Set the main function as the entry point of the application.
if __name__ == '__main__':
    main()
