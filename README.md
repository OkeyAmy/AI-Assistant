# Personal AI Assistant

Welcome to the **Personal AI Assistant** project! This application is designed to help users interact with AI to manage and respond to messages. It integrates with various messaging platforms, processes documents, and provides AI-driven responses.

## Features

- **Interactive Chat Interface**: Communicate with an AI assistant that provides relevant responses to user queries.
- **Document Upload and Processing**: Upload text, PDF, and DOCX files to create a searchable vector store.
- **AI-Driven Responses**: Receive contextually relevant responses based on user input and document content.
- **Configurable Settings**: Toggle options to allow external information and control document processing.

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Pip** (Python package installer)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/okeyamy/your-repository-name.git
   cd your-repository-name
2. **Set Up a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
3. **Configure Environment Variables**
   Create a .env file in the root directory with the following content:

   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
4. **Run the Application**
   Start the Streamlit application:

   ```bash
   streamlit run app.py



## Usage

1. **Access the Application**

Open your web browser and navigate to http://localhost:8501 to start using the Personal AI Assistant.

2. **Interact with the AI**

Type your message in the input field to ask questions or interact with the AI.
Upload documents via the sidebar. Supported formats include .txt, .pdf, and .docx.

3. **Settings**

Use the checkbox in the sidebar to toggle Allow external information on or off.

