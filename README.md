🏥 Medical AI Assistant - RAG Chatbot
<div align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version"> <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" alt="Streamlit"> <img src="https://img.shields.io/badge/PyTorch-2.0+-orange.svg" alt="PyTorch"> <img src="https://img.shields.io/badge/Transformers-4.0+-yellow.svg" alt="Transformers"> <img src="https://img.shields.io/badge/License-Educational-green.svg" alt="License"> </div> <div align="center"> <h3>🤖 Intelligent Medical Assistant powered by IBM Granite & RAG Technology</h3> <p>A sophisticated Streamlit application providing context-aware medical information with user authentication and personalized search history.</p> </div>
🌟 Features
<table> <tr> <td width="50%">
🤖 Advanced AI Capabilities
🧠 IBM Granite Model Integration
ibm-granite/granite-3.3-2b-instruct
🔍 RAG Technology
Retrieval-Augmented Generation
🎯 Semantic Search
Sentence transformers matching
⚡ Response Caching
Optimized performance
</td> <td width="50%">
👥 User Management
🔐 Secure Authentication
Login & registration system
👤 Personalized Experience
User-specific preferences
📚 Search History
Previous query tracking
⚙️ User Profiles
Profile management
</td> </tr> <tr> <td width="50%">
🏥 Medical Focus
📖 Medical Knowledge Base
Trained on medical Q&A dataset
🩺 Professional Responses
Medically accurate information
⚠️ Safety Disclaimers
Automatic medical warnings
🔍 Context Transparency
Source reference display
</td> <td width="50%">
⚡ Performance Optimizations
🖥️ CPU/GPU Flexibility
Automatic hardware detection
🧠 Memory Management
Efficient resource usage
💾 Model Caching
Streamlit optimization
🔄 Async Support
Responsive UI experience
</td> </tr> </table>
🚀 Quick Start
Prerequisites
📋 System Requirements
├── Python 3.8+
├── RAM: 8GB minimum (16GB recommended)
├── Storage: 10GB free space
└── Optional: CUDA GPU for acceleration
🔧 Installation
Clone the repository
bash
git clone https://github.com/yourusername/medical-ai-assistant.git
cd medical-ai-assistant
Install dependencies
bash
pip install -r requirements.txt
Set up your environment
bash
# Create requirements.txt with:
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.0.0
sentence-transformers>=2.2.0
transformers>=4.21.0
torch>=1.12.0
nest-asyncio>=1.5.0
Prepare your dataset
python
# Update in main script:
data_path = "path/to/your/medical_dataset.csv"
Run the application
bash
streamlit run app.py
🌐 Access at: http://localhost:8501

📊 Dataset Format
Your medical Q&A dataset should follow this structure:

csv
short_question,response
"What are the symptoms of diabetes?","Common symptoms of diabetes include increased thirst, frequent urination..."
"How is hypertension treated?","Hypertension treatment typically involves lifestyle modifications and medications..."
"What causes migraine headaches?","Migraine headaches can be triggered by various factors including stress..."
<details> <summary>📋 <strong>Supported Column Names</strong></summary>
Question Columns	Answer Columns
short_question	response
question	answer
query	reply
input	output
prompt	-
</details>
🏗️ Architecture Overview
mermaid
graph TB
    A[User Query] --> B[Authentication Check]
    B --> C[Embedding Model]
    C --> D[Similarity Search]
    D --> E[Context Retrieval]
    E --> F[IBM Granite Model]
    F --> G[Response Generation]
    G --> H[Medical Disclaimer]
    H --> I[User History]
    I --> J[Final Response]
    
    subgraph "RAG Pipeline"
        C
        D
        E
        F
    end
    
    subgraph "User Management"
        B
        I
    end
🔧 Core Components
Component	Technology	Purpose
Embedding Model	all-MiniLM-L6-v2	Semantic question matching
Language Model	IBM Granite 3.3-2B	Response generation
Vector Search	Cosine Similarity	Context retrieval
User Backend	SQLite/Database	Authentication & history
Caching System	Multi-level	Performance optimization
🎯 Usage Guide
🏁 Getting Started
🔐 Authentication
→ Register new account or login
→ Access personalized dashboard
⚙️ Configuration (Sidebar)
python
# Retrieval Settings
top_k = 1-5          # Similar questions to retrieve
temperature = 0.1-1.0 # Response creativity
max_new_tokens = 100-500 # Response length

# Performance Settings
force_cpu = True/False    # CPU vs GPU mode
use_smaller_model = True/False # Model size option
💬 Chat Interface
→ Type medical questions
→ View AI responses with context
→ Check retrieved references
→ Review search history
📱 Interface Preview
┌─────────────────────────────────────────┐
│  🏥 Medical AI Assistant                │
├─────────────────────────────────────────┤
│  👤 User: "What are diabetes symptoms?" │
│  🤖 Bot: "Common symptoms include..."   │
│       📋 View retrieved context ▼       │
│         Reference 1: (Similarity: 0.89) │
│         Q: What are signs of diabetes?  │
│         A: Signs include frequent...    │
└─────────────────────────────────────────┘
⚡ Performance & Optimization
🖥️ System Configurations
<div align="center">
Configuration	RAM Usage	Load Time	Response Time
GPU Mode	~6GB	~2-3 min	~2-5 sec
CPU Mode	~4GB	~5-8 min	~10-20 sec
Small Model	~3GB	~1-2 min	~5-10 sec
</div>
🚀 Optimization Features
python
# Memory Management
✅ Automatic garbage collection
✅ GPU memory clearing
✅ Model parameter optimization
✅ Efficient tokenization

# Speed Enhancements  
✅ Response caching
✅ Context caching
✅ Streamlit resource caching
✅ Batch processing
🛠️ Troubleshooting
🚨 Common Issues & Solutions
<details> <summary>🐛 <strong>Model Loading Errors</strong></summary>
bash
# Error: CUDA out of memory
❌ Error loading model: CUDA out of memory

# Solutions:
✅ Enable "Force CPU mode" in sidebar
✅ Use "smaller model" option  
✅ Restart application
✅ Close other GPU applications
</details> <details> <summary>📦 <strong>Import Errors</strong></summary>
bash
# Error: Missing dependencies
❌ Error importing sentence_transformers

# Solution:
✅ pip install sentence-transformers transformers torch
✅ pip install --upgrade streamlit
✅ Check Python version (3.8+ required)
</details> <details> <summary>📁 <strong>Dataset Issues</strong></summary>
bash
# Error: Dataset not found
❌ Error loading dataset: No such file or directory

# Solutions:
✅ Update data_path in code
✅ Check CSV format (see Dataset Format section)
✅ Ensure proper column names
✅ Verify file permissions
</details> <details> <summary>⚡ <strong>Performance Issues</strong></summary>
python
# Slow responses?
✅ Reduce max_new_tokens (300 → 150)
✅ Lower top_k value (5 → 2)  
✅ Enable CPU mode for stability
✅ Clear browser cache
✅ Restart Streamlit server
</details>
📞 Getting Help
bash
# Debug Mode
streamlit run app.py --logger.level=debug

# Check System Info
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
🔧 Customization
🤖 Using Different Models
python
# In load_granite_model() function:
model_options = {
    "small": "ibm-granite/granite-3.3-2b-instruct",
    "medium": "ibm-granite/granite-3.3-8b-instruct", 
    "large": "ibm-granite/granite-3.3-20b-instruct"
}
🎨 UI Customization
python
# Custom CSS styling
st.markdown("""
<style>
    .stApp { 
        background-color: #f0f8ff; 
    }
    .chat-message.user { 
        background-color: #e1f5fe; 
    }
    .chat-message.bot { 
        background-color: #f3e5f5; 
    }
</style>
""", unsafe_allow_html=True)
📊 Dataset Customization
python
# Custom column mapping
column_mapping = {
    'questions': ['question', 'query', 'input'],
    'answers': ['answer', 'response', 'output']
}
🤝 Contributing
We welcome contributions! Here's how you can help:

<div align="center">
🐛 Bug Reports	💡 Feature Requests	📝 Documentation	🧪 Testing
Report issues	Suggest improvements	Improve docs	Add test cases
</div>
🔀 Development Workflow
bash
# 1. Fork & Clone
git clone https://github.com/yourusername/medical-ai-assistant.git
cd medical-ai-assistant

# 2. Create Feature Branch  
git checkout -b feature/amazing-feature

# 3. Make Changes & Test
python -m pytest tests/
streamlit run app.py

# 4. Commit & Push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# 5. Create Pull Request
# → Open PR on GitHub with description
📋 Contribution Guidelines
✅ Follow PEP 8 style guidelines
✅ Add docstrings for new functions
✅ Include tests for new features
✅ Update documentation as needed
✅ Test on both CPU and GPU modes
📄 License & Legal
<div align="center">
Component	License	Usage
Application Code	MIT License	Open source
IBM Granite Model	IBM License	Check terms
Hugging Face Models	Apache 2.0	Commercial OK
Medical Content	Educational	Non-commercial
</div>
⚖️ Important Legal Notes
📚 Educational Purpose Only
├── Not for commercial medical advice
├── Requires professional medical consultation  
├── No liability for medical decisions
└── Compliance with local healthcare regulations
⚠️ Medical Disclaimer
<div align="center"> <table> <tr> <td align="center"> <h3>🏥 IMPORTANT MEDICAL DISCLAIMER</h3> <p><strong>This AI assistant is for educational and informational purposes only.</strong></p> <p>❌ Not a substitute for professional medical advice<br> ❌ Not for diagnosis or treatment decisions<br> ❌ Not for emergency medical situations</p> <p>✅ Always consult qualified healthcare professionals<br> ✅ Verify all medical information independently<br> ✅ Use only as a learning supplement</p> </td> </tr> </table> </div>
📞 Support & Community
<div align="center">
Show Image
Show Image
Show Image

</div>
🆘 Getting Support
📖 Documentation: Check this README first
🐛 Issues: Create GitHub Issue
💬 Discussions: GitHub Discussions
📧 Contact: Open an issue for direct support
🏷️ Project Info
yaml
Version: 1.0.0
Last Updated: June 2025
Python Compatibility: 3.8+
Streamlit Compatibility: 1.28+
Platform: Windows, macOS, Linux
Status: Active Development
<div align="center"> <h3>⭐ If this project helped you, please give it a star! ⭐</h3> <p> <a href="https://github.com/yourusername/medical-ai-assistant"> <img src="https://img.shields.io/badge/⭐-Star%20on%20GitHub-yellow?style=for-the-badge" alt="Star on GitHub"> </a> </p> <p><em>Built with ❤️ for the medical AI community</em></p> </div>
