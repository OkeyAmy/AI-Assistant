import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
import os
import tempfile
import pickle

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import GooglePalmEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain, question_answering

# Load environment variables
load_dotenv()

# Set Google API key
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def init():
    """
    Initialize Streamlit page configuration and settings.
    """
    st.set_page_config(
        page_title="Your Personal AI Assistant",
        page_icon="random",
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Report a bug': "mailto:amaobiokeoma@gmail.com",
            'About': """ 
            My name is Okey Amy, an ML Engineer passionate about AI and all things tech.
            I love building tools that make life easier and smarter. With a background in machine 
            learning and experience in creating interactive AI assistants, I'm excited to share my latest 
            project—a personal AI assistant that helps with general knowledge and document analysis.
            """
        }
    )

def load_document(document_file):
    """
    Load the uploaded document, create embeddings, and save the FAISS vector store.
    """
    if document_file is not None:
        # Determine the file type and use the appropriate loader
        loaders = {
            'txt': TextLoader,
            'pdf': PyPDFLoader,
            'docx': Docx2txtLoader
        }
        file_type = document_file.name.split('.')[-1].lower()
        loader = loaders.get(file_type)

        if loader is None:
            st.error('Unsupported file format.')
            return None

        # Save the uploaded document to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_type}') as temp_file:
            temp_file.write(document_file.read())
            tmp_file_path = temp_file.name

        # Load the document and create the FAISS vector store
        try:
            loader_instance = loader(tmp_file_path)
            documents = loader_instance.load_and_split()
            # Use GooglePalmEmbeddings if needed, or HuggingFaceEmbeddings as a fallback
            embedding = GooglePalmEmbeddings(show_progress_bar=True) if os.getenv('GOOGLE_API_KEY') else HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
            vectorstore = FAISS.from_documents(documents=documents, embedding=embedding, show_progress_bar=True)

            # Save the vector store to a temporary file
            vectorstore_path = tempfile.mktemp(suffix='.pkl')
            with open(vectorstore_path, 'wb') as f:
                pickle.dump(vectorstore, f)

            return vectorstore_path
        except Exception as e:
            st.error(f'Error processing document: {e}')
            return None

def main():
    init()
    
    st.header('Your Personal AI Assistant 🤖')

    # Initialize the chat model
    chat = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Add a system message if not present
    system_instruction = SystemMessage(content=''' 
        You are a resourceful AI that assists users with their queries by providing accurate information. 
        If you don't have the information directly, you will automatically search for and provide a relevant 
        website link that may contain the answer, without stating your limitations. Avoid using words like 
        "however", "moreover", and similar in your responses.
    ''')
    if not st.session_state.messages or not isinstance(st.session_state.messages[0], SystemMessage):
        st.session_state.messages.insert(0, system_instruction)

    # User input and document upload
    user_input = st.chat_input("Type your message here...")
    upload_document = st.sidebar.file_uploader('Upload your txt, pdf, or docx document file here')

    # Allow external information (toggle in sidebar)
    st.sidebar.header('Settings')
    allow_external = st.sidebar.checkbox("Allow external information", value=False)

    # Load the document and create a vector store if not already done
    if upload_document and 'vectorstore_path' not in st.session_state:
        with st.spinner("Loading your document..."):
            st.session_state.vectorstore_path = load_document(upload_document)
            st.session_state.retriever = None

    # Load the retriever from the saved vector store
    if 'vectorstore_path' in st.session_state and st.session_state.retriever is None:
        try:
            with open(st.session_state.vectorstore_path, 'rb') as f:
                vectorstore = pickle.load(f)
                st.session_state.retriever = vectorstore.as_retriever()
        except Exception as e:
            st.error(f'Error loading vector store: {e}')
            st.session_state.retriever = None

    # Process user input and provide responses
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        if 'retriever' in st.session_state and not allow_external:
            retriever = st.session_state.retriever
            qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=chat,
                chain_type="stuff",
                retriever=retriever
            )
            try:
                matching_results = retriever.get_relevant_documents(user_input)
                chain = question_answering.load_qa_chain(chat, chain_type='stuff')
                response = chain.run(input_documents=matching_results, question=user_input)
            except Exception as e:
                response = chat(st.session_state.messages).content
        else:
            # Filter out SystemMessage from the chat history
            filtered_messages = [msg for msg in st.session_state.messages if not isinstance(msg, SystemMessage)]
            with st.spinner('Thinking...'):
                response = chat(filtered_messages).content

        with st.spinner('Thinking...'):
            st.session_state.messages.append(AIMessage(content=response))

    # Display the chat history
    for i, msg in enumerate(st.session_state.get('messages', [])):
        if isinstance(msg, HumanMessage):
            message(msg.content, is_user=True, key=str(i) + '_user')
        elif isinstance(msg, AIMessage):
            message(msg.content, is_user=False, key=str(i) + '_ai')

if __name__ == "__main__":
    main()
