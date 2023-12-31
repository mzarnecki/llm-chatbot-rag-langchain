# RAG

LLM (Large Language Model) based chat bot using RAG (Retrieval-augmented generation) and langchain.
Chat bot has access to trade register documents related to Ticos Systems company and is able to answer any question related to this company not only giving answer but also referencing the source document.

as language model HuggingFaceH4/zephyr-7b-beta was used
to run chat in browser streamlit library was used

based on article https://medium.com/mlearning-ai/create-a-chatbot-in-python-with-langchain-and-rag-85bfba8c62d2

## Usage
1. before run install needed libraries as below:
`pip install langchain faiss-cpu openai tiktoken InstructorEmbedding sentence-transformers streamlit python-dotenv pandas trafilatura`

2. create .env file with hugiingface token environment variable 
`HUGGINGFACEHUB_API_TOKEN=your_token_from_huggingface.com`

3. run script with command:
`streamlit run index.py`