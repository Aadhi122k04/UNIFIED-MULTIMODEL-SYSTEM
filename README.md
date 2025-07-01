📄 QueryVox - Document Analyzer with RAG
A simple AI-powered document analyzer that can read PDFs, images, and audio files, then answer questions about them using voice responses.
🤖 What does it do?

Upload PDFs, images, or audio files
Ask questions about the content
Get answers in both text and voice format
Keep track of your previous questions and answers
Download audio responses for offline use

🛠️ Technologies Used

Python - Main programming language
Streamlit - Web interface
LangChain - For RAG (Retrieval Augmented Generation)
Hugging Face - AI models for text processing
Google Text-to-Speech - Convert text to voice
ChromaDB - Store document embeddings
pyPdfLoader - Read PDF files

📋 Requirements

Python 3.8+
Internet connection
At least 8GB RAM

🚀 How to Run
1. Download the project
bashgit clone https://github.com/yourusername/queryvox-rag.git
cd queryvox-rag
2. Install required packages
bashpip install -r requirements.txt
3. Run the app
bashstreamlit run app.py
4. Open your browser
Go to http://localhost:8501
📦 Required Packages
Create a requirements.txt file with:
streamlit
langchain
sentence-transformers
chromadb
PyPDF2
google-cloud-texttospeech
transformers
torch
💻 How to Use

Choose file type: Select PDF, Image, or Audio
Upload file: Drag and drop or browse for your file
Wait for processing: The system will analyze your document
Ask questions: Type your question in the text box
Get answers: Read the text response or listen to the audio
Check history: See all your previous questions and answers

📁 Project Structure
queryvox-rag/
├── app.py              # Main application file
├── requirements.txt    # All required Python packages
├── README.md          # This file
└── docs/              # Documentation folder
🔧 Features
Document Processing

Reads PDF files and extracts text
Processes images using OCR
Transcribes audio files to text

Question Answering

Uses RAG to find relevant information
Generates context-aware answers
Converts answers to speech
