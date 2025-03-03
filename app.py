# Import Langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import RecursiveCharacterSetTextSplitter
# Bring in streamLit for UI dev
# Bring in streamLit for UI dev
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()


# Set up LLM
llm = ChatGroq(
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    max_tokens=6000,
    max_retries=2,
    stop_sequences=["."]
        )

# Set up the app title
st.title('Ask me')

# Set up a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
prompt = st.chat_input('Pass your prompt here')

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    user_prompt = PromptTemplate.from_template(
        """
                {prompt}
        """
    )
    chain_user = user_prompt | llm
    response = chain_user.invoke({'prompt': prompt})
    st.code(response.content, language='markdown')
    # st.chat_message('system').markdown(response.content)
    st.session_state.messages.append({'role': 'system', 'content': response.content})