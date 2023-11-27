There are two components to the vastuGPT. One is the RAG part and the other the Baseline response. Due to technical issues, the Retrieval part is built on flask and the output is routed to streamlit app. The Baseline response from OpenAI and the Generator part of the RAG module is built in the vastuGPT.py file. To Run the app, follow the below steps: 

1. Save the py files, vastu_shastra_processed.doc and the api_keys.json files in one single folder
2. Open 2 commnad prompts and make sure you are in the folder that contains the code files 
3. On one of the command prompts, run the command 
	python flask_rag.py 
4. On the other command prompt, run the command 
	streamlit run vastuGPT.py 
The vastuGPT application which is locally deployed on streamlit will open




