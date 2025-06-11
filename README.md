Medical AI Assistant - RAG Chatbot
A sophisticated medical AI assistant powered by IBM Granite model with Retrieval-Augmented Generation (RAG) technology. This Streamlit application provides intelligent medical information retrieval with user authentication and personalized search history.

Features
🤖 Advanced AI Capabilities
IBM Granite Model Integration: Utilizes the powerful ibm-granite/granite-3.3-2b-instruct model for generating medical responses
RAG Technology: Combines retrieval-based and generative approaches for accurate, context-aware responses
Semantic Search: Uses sentence transformers for intelligent question matching and context retrieval
Response Caching: Optimized performance with intelligent caching mechanisms
👥 User Management
User Authentication: Secure login and registration system
Personalized Experience: User-specific search history and preferences
Search History: Tracks and leverages previous queries for improved responses
User Profiles: Dedicated user profile management
🏥 Medical Focus
Medical Knowledge Base: Trained on medical Q&A dataset
Professional Responses: Generates medically accurate and professional responses
Safety Disclaimers: Automatically includes medical disclaimers for safety
Context Retrieval: Shows relevant medical context sources for transparency
⚡ Performance Optimizations
CPU/GPU Flexibility: Supports both CPU and GPU execution with automatic fallback
Memory Management: Efficient memory usage with garbage collection and torch optimizations
Model Caching: Streamlit caching for faster subsequent loads
Async Support: Nested asyncio support for improved responsiveness
Installation
Prerequisites
Python 3.8 or higher
At least 8GB RAM (16GB recommended for optimal performance)
Optional: CUDA-compatible GPU for faster inference
Required Dependencies
bash
pip install streamlit pandas numpy scikit-learn sentence-transformers transformers torch nest-asyncio
Detailed Installation
Clone or download the application files
Install Python dependencies:
bash
pip install -r requirements.txt
Prepare your dataset:
Ensure you have a CSV file with medical Q&A data
Update the data_path variable in the code to point to your dataset
Expected columns: short_question/question and response/answer
Set up user management:
Ensure the user_management.py module is available
This should include functions for user authentication and history management
Usage
Starting the Application
bash
streamlit run app.py
The application will be available at http://localhost:8501

First Time Setup
Register/Login: Create an account or log in to access the chatbot
Model Loading: The system will automatically load the required models on first use
Dataset Processing: Your medical Q&A dataset will be processed and embeddings created
Start Chatting: Begin asking medical questions once setup is complete
Configuration Options
The sidebar provides several configuration options:

Retrieval Settings:
top_k: Number of similar questions to retrieve (1-5)
temperature: Response creativity (0.1-1.0)
max_new_tokens: Maximum response length (100-500)
Performance Settings:
Force CPU mode for systems without compatible GPU
Model optimization options
Dataset Format
Your CSV dataset should contain medical Q&A pairs with the following structure:

csv
short_question,response
"What are the symptoms of diabetes?","Common symptoms of diabetes include..."
"How is hypertension treated?","Hypertension treatment typically involves..."
Supported column names:

Questions: short_question, question, query, input, prompt
Answers: response, answer, reply, output
Architecture
Core Components
Embedding Model: all-MiniLM-L6-v2 for semantic question matching
Language Model: IBM Granite 3.3-2B Instruct for response generation
Vector Database: NumPy-based similarity search with cosine similarity
User Management: SQLite/database backend for user authentication
Caching System: Multi-level caching for embeddings and responses
RAG Pipeline
Query Processing: User question is encoded using sentence transformers
Context Retrieval: Similar questions found using cosine similarity
Context Formatting: Relevant Q&A pairs formatted as context
Response Generation: Granite model generates response using retrieved context
Post-processing: Response cleaning and medical disclaimer addition
Performance Optimization
Memory Management
Automatic garbage collection
GPU memory clearing with torch.cuda.empty_cache()
Model parameter optimization (half precision on GPU)
Efficient tokenization with padding and truncation
Speed Optimizations
Response and context caching
Streamlit resource caching for models
Batch processing optimizations
Flash attention support (when available)
System Requirements
Minimum Requirements:

8GB RAM
10GB free disk space
Python 3.8+
Recommended Requirements:

16GB RAM
NVIDIA GPU with 8GB+ VRAM
SSD storage
Python 3.9+
Troubleshooting
Common Issues
Model Loading Errors:

Error loading model: CUDA out of memory
Solution: Enable "Force CPU mode" in sidebar settings
Alternative: Use smaller model option
Import Errors:

Error importing sentence_transformers
Solution: Install missing dependencies:
bash
pip install sentence-transformers transformers torch
Dataset Loading Issues:

Error loading dataset: No such file or directory
Solution: Update the data_path variable to point to your CSV file
Performance Issues:

Reduce max_new_tokens for faster responses
Lower top_k value for less context retrieval
Enable CPU mode if GPU causes instability
Error Messages
The application provides detailed error messages and suggestions:

Memory issues → CPU mode recommendation
Missing dependencies → Installation commands
Model loading failures → Alternative approaches
Security Considerations
User passwords should be properly hashed (implement in user_management.py)
Session management with secure token handling
Input sanitization for SQL injection prevention
Rate limiting for API calls (recommended for production)
Customization
Adding New Models
To use different models, modify the model_name variable in load_granite_model():

python
model_name = "your-preferred-model"
Custom Datasets
Update the dataset path and column mapping in the configuration section:

python
data_path = "path/to/your/dataset.csv"
UI Customization
Modify the CSS in the st.markdown() section for custom styling:

python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
Contributing
Fork the repository
Create a feature branch
Implement your changes
Add appropriate tests
Submit a pull request
License
This project is intended for educational and research purposes. Please ensure compliance with:

IBM Granite model license terms
Hugging Face model licenses
Medical information usage regulations
Disclaimer
Important Medical Disclaimer: This AI assistant is designed for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

Support
For issues and questions:

Check the troubleshooting section above
Verify all dependencies are properly installed
Ensure your dataset format matches the expected structure
Check system requirements and available resources
Version: 1.0
Last Updated: June 2025
Compatible with: Python 3.8+, Streamlit 1.28+


