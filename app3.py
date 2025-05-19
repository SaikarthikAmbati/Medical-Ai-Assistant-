import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import time
import warnings
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
try:
    nest_asyncio.apply()
except:
    pass

# Ensure there's a running event loop
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Suppress TensorFlow and MessageFactory warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Disable TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# Import the rest of the libraries with error handling
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("Error importing scikit-learn. Please install it with: pip install scikit-learn")
    st.stop()

# Handle torch import carefully
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    st.warning("PyTorch not available. Will try to run in CPU-only mode.")

# Try to import sentence_transformers safely
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    st.error("Error importing sentence_transformers. Please install it: pip install sentence-transformers")

# Try to import transformers safely
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    st.error("Error importing transformers. Please install it: pip install transformers")

# Only attempt to import these if torch is available
if HAS_TORCH:
    # Handle asyncio-related imports only if torch is available
    try:
        import asyncio
        import nest_asyncio
        
        # Apply nest_asyncio to allow nested event loops 
        try:
            nest_asyncio.apply()
        except:
            pass
        
        # Ensure there's a running event loop
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
    except ImportError:
        pass  # Asyncio not required if not using certain features

# Check if we can run the app based on available dependencies
if not HAS_SENTENCE_TRANSFORMERS or not HAS_TRANSFORMERS:
    st.error("Missing critical dependencies. Please install all required packages.")
    st.code("pip install streamlit pandas sentence-transformers scikit-learn transformers torch nest-asyncio", language="bash")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="Granite RAG Chatbot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
        margin-right: 1rem;
    }
    .chat-message .message {
        flex-grow: 1;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .info-box {
        background-color: #2b313e;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Add title and description
st.title("Medical Ai assistant")
st.markdown("Ask questions and get responses powered by IBM Granite model with RAG technology")

# Import user management
from user_management import (
    initialize_session_state,
    show_login_form,
    show_registration_form,
    show_user_profile
)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'model_loaded' not in st.session_state:
    st.session_state['model_loaded'] = False
if 'embedding_model' not in st.session_state:
    st.session_state['embedding_model'] = None
if 'granite_model' not in st.session_state:
    st.session_state['granite_model'] = None
if 'tokenizer' not in st.session_state:
    st.session_state['tokenizer'] = None
if 'question_embeddings' not in st.session_state:
    st.session_state['question_embeddings'] = None
if 'questions' not in st.session_state:
    st.session_state['questions'] = []
if 'answers' not in st.session_state:
    st.session_state['answers'] = []
if 'loading_progress' not in st.session_state:
    st.session_state['loading_progress'] = None
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'process_input' not in st.session_state:
    st.session_state['process_input'] = False
if 'response_cache' not in st.session_state:
    st.session_state['response_cache'] = {}

# Initialize user management
initialize_session_state()

# Sidebar for configuration
with st.sidebar:
    # Show user profile if logged in
    if st.session_state.is_logged_in:
        show_user_profile()
    else:
        # Show login/registration forms
        if st.session_state.show_registration:
            show_registration_form()
        else:
            show_login_form()
    
    st.header("Configuration")
    
    # Only show configuration if logged in
    if st.session_state.is_logged_in:
        # Dataset upload
        uploaded_file = "C:/Users/saika/Desktop/testing rtp/train_data_chatbot.csv"
        data_path = "C:/Users/saika/Desktop/testing rtp/train_data_chatbot.csv"
        
        # Model parameters
        st.subheader("Model Parameters")
        top_k = st.slider("Number of similar questions to retrieve", 1, 5, 2)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
        max_new_tokens = st.slider("Maximum new tokens in response", 100,500,300)
        
        # Option to use a smaller model or offload to CPU
        use_smaller_model = False
        force_cpu = True
        
        # Button to load/reload model
        if st.button("Load/Reload Model", key="load_model"):
            st.session_state['model_loaded'] = False

# Define functions for the chatbot
@st.cache_resource
def load_embedding_model():
    """Load the embedding model - this gets cached by Streamlit"""
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_dataset(file_path=None, uploaded_file=None):
    """Load dataset from either uploaded file or path"""
    try:
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_csv(file_path)
        
        st.success(f"Dataset loaded with {len(data)} examples")
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

def prepare_qa_data(data):
    """Extract questions and answers from dataframe"""
    # Make sure we have a clean dataframe
    if data is None or len(data) == 0:
        st.error("Dataset is empty or invalid")
        return [], []
        
    # Handle various column naming conventions
    if 'short_question' in data.columns:
        questions = data['short_question'].tolist()
    elif 'question' in data.columns:
        questions = data['question'].tolist()
    else:
        # Try to find a likely question column
        potential_question_cols = [col for col in data.columns if any(term in col.lower() 
                                for term in ['question', 'query', 'input', 'prompt'])]
        
        if potential_question_cols:
            question_col = potential_question_cols[0]
        else:
            # Default to the first column if no obvious question column is found
            question_col = data.columns[0]
        
        st.info(f"Using '{question_col}' as the question column")
        questions = data[question_col].tolist()
    
    # Determine answer column
    if 'response' in data.columns:
        answers = data['response'].tolist()
    elif 'answer' in data.columns:
        answers = data['answer'].tolist()
    else:
        # Try to find a likely answer column
        potential_answer_cols = [col for col in data.columns if any(term in col.lower() 
                                for term in ['answer', 'response', 'reply', 'output'])]
        
        if potential_answer_cols:
            answer_col = potential_answer_cols[0]
        else:
            # Default to the second column if no obvious answer column is found
            answer_col = data.columns[1] if len(data.columns) > 1 else data.columns[0]
        
        st.info(f"Using '{answer_col}' as the answer column")
        answers = data[answer_col].tolist()
    
    return questions, answers

def create_embeddings(questions, embedding_model):
    """Create embeddings for questions"""
    embedding_file = "question_embeddings.npy"
    
    # Check if embeddings already exist
    if os.path.exists(embedding_file):
        st.info(f"Loading existing embeddings from {embedding_file}")
        question_embeddings = np.load(embedding_file)
        st.success(f"Loaded embeddings shape: {question_embeddings.shape}")
    else:
        progress_text = "Creating embeddings for the knowledge base..."
        progress_bar = st.progress(0)
        
        # Create embeddings with progress updates
        question_embeddings = embedding_model.encode(
            questions, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Save embeddings for future use
        np.save(embedding_file, question_embeddings)
        st.success(f"Embeddings created and saved to {embedding_file}")
        progress_bar.empty()
    
    return question_embeddings

def load_granite_model(use_smaller_model=False, force_cpu=True):
    """Load the IBM Granite model with improved error handling and optimizations"""
    # Select model based on system capabilities
    if use_smaller_model:
        model_name = "ibm-granite/granite-3.3-2b-instruct"  # Smaller model
    else:
        model_name = "ibm-granite/granite-3.3-2b-instruct"  # Default model
    
    loading_text = st.empty()
    loading_text.text(f"Loading {model_name}. This may take several minutes...")
    progress_bar = st.progress(0)
    
    try:
        # Free up memory before loading the model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Configure device map and optimization settings
        if force_cpu:
            device_map = "cpu"
            torch_dtype = torch.float32
            loading_text.text(f"Loading {model_name} on CPU with optimizations...")
        else:
            device_map = "auto"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        progress_bar.progress(10)
        loading_text.text("Initializing tokenizer...")
        
        # Load tokenizer with optimizations
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=True,
                model_max_length=512
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            st.error(f"Error loading tokenizer: {str(e)}")
            return None, None
        
        progress_bar.progress(30)
        loading_text.text("Loading model with optimizations...")
        
        # Load model with optimizations
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="offload_folder",
                use_cache=True,  # Enable KV cache
                use_flash_attention_2=True if not force_cpu else False,  # Enable flash attention if not on CPU
            )
            
            # Enable model optimizations
            model.eval()  # Set to evaluation mode
            if not force_cpu and torch.cuda.is_available():
                model = model.half()  # Use half precision if on GPU
            
            progress_bar.progress(90)
            loading_text.text("Model loaded with optimizations!")
            
        except Exception as e:
            progress_bar.progress(40)
            loading_text.text(f"Optimized loading failed: {str(e)}\nTrying with safetensors loading...")
            
            # Try with safetensors
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    use_safetensors=True,
                    trust_remote_code=True,
                    use_cache=True,
                )
                model.eval()
                if not force_cpu and torch.cuda.is_available():
                    model = model.half()
                progress_bar.progress(90)
                loading_text.text("Model loaded with safetensors!")
            except Exception as e:
                st.error(f"Failed to load model with safetensors: {str(e)}")
                return None, None
        
        # Final cleanup and completion
        progress_bar.progress(100)
        loading_text.text("Performing final model setup...")
        
        # Sleep briefly to allow UI to update
        time.sleep(0.5)
        
        # Clear progress indicators
        progress_bar.empty()
        loading_text.empty()
        
        return model, tokenizer
        
    except Exception as e:
        progress_bar.empty()
        loading_text.empty()
        st.error(f"Error loading model: {str(e)}")
        
        # Suggest solutions based on the error
        if "CUDA out of memory" in str(e):
            st.warning("GPU memory issue detected. Try checking 'Force CPU mode' and 'Use smaller model' options in the sidebar.")
        elif "disk space" in str(e).lower():
            st.warning("Disk space issue detected. Free up some disk space and try again.")
        else:
            st.warning("Try restarting the application or using a smaller model.")
            
        return None, None

def get_relevant_context(query, top_k=2):
    """Find relevant context from the dataset with caching"""
    # Check cache first
    cache_key = f"context_{query}_{top_k}"
    if cache_key in st.session_state['response_cache']:
        return st.session_state['response_cache'][cache_key]
    
    # Encode the query
    query_embedding = st.session_state['embedding_model'].encode([query], convert_to_numpy=True)[0]
    
    # Calculate similarity with all questions in the dataset
    similarities = cosine_similarity([query_embedding], st.session_state['question_embeddings'])[0]
    
    # Get top k similar questions and their answers
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Collect the most relevant QA pairs as context
    contexts = []
    for idx in top_indices:
        contexts.append({
            "question": st.session_state['questions'][idx],
            "answer": st.session_state['answers'][idx],
            "similarity": similarities[idx]
        })
    
    # Cache the result
    st.session_state['response_cache'][cache_key] = contexts
    return contexts

def generate_response(prompt, max_new_tokens=300, temperature=0.7):
    """Generate a response using the Granite model with optimizations"""
    # Check cache first
    cache_key = f"response_{prompt}_{max_new_tokens}_{temperature}"
    if cache_key in st.session_state['response_cache']:
        return st.session_state['response_cache'][cache_key]
    
    tokenizer = st.session_state['tokenizer']
    model = st.session_state['granite_model']
    
    if model is None or tokenizer is None:
        return "Model not properly loaded. Please reload the model and try again."
    
    # Optimize input processing with batch size 1
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )
    
    try:
        # Move inputs to the correct device if model is on CUDA
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad(), torch.cuda.amp.autocast() if torch.cuda.is_available() else nullcontext():
            # Optimize generation parameters for speed
            output = model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                early_stopping=True,
                no_repeat_ngram_size=3  # Prevent repetition
            )
        
        # Optimize decoding
        response = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Clean up the response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        # Cache the result
        st.session_state['response_cache'][cache_key] = response
        return response
        
    except Exception as e:
        return f"I apologize, but I encountered an error while generating a response. Error details: {str(e)}"

def generate_rag_response(query, top_k=2, max_new_tokens=300, temperature=0.7):
    """Retrieve relevant context and generate a response using the Granite model with caching and user history"""
    # Check cache first
    cache_key = f"rag_{query}_{top_k}_{max_new_tokens}_{temperature}"
    if cache_key in st.session_state['response_cache']:
        return st.session_state['response_cache'][cache_key]
    
    # Get relevant context
    contexts = get_relevant_context(query, top_k=top_k)
    
    # If user is logged in, get similar queries from their history
    user_history_context = ""
    if st.session_state.is_logged_in:
        similar_queries = st.session_state.user_management.get_similar_queries(
            st.session_state.user['user_id'], query)
        if similar_queries:
            user_history_context = "\n\nPrevious related queries from your history:\n"
            for item in similar_queries:
                user_history_context += f"- Q: {item['query']}\n  A: {item['response']}\n"
    
    # Format context for the model
    formatted_context = ""
    for idx, ctx in enumerate(contexts):
        formatted_context += f"Reference {idx+1}:\nQuestion: {ctx['question']}\nAnswer: {ctx['answer']}\n\n"
    
    # Create optimized prompt with user history
    prompt = f"""<instruction>You are a medical AI assistant. Use ONLY the provided medical information:

{formatted_context}

{user_history_context}

Question: {query}

Provide a concise, professional medical response based on the above information.</instruction>"""
    
    # Generate response
    full_response = generate_response(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # Clean up the response
    if "</instruction>" in full_response:
        answer = full_response.split("</instruction>")[-1].strip()
    else:
        answer = full_response.replace(prompt, "").strip()
    
    # Add medical disclaimer if not present
    if "disclaimer" not in answer.lower():
        answer += "\n\nDisclaimer: This information is provided for educational purposes only and should not be considered as medical advice. Please consult with a qualified healthcare professional for proper medical guidance."
    
    # Save to user history if logged in
    if st.session_state.is_logged_in:
        try:
            print(f"Saving history for user {st.session_state.user['user_id']}")
            success = st.session_state.user_management.save_search_history(
                st.session_state.user['user_id'],
                query,
                answer,
                contexts
            )
            if not success:
                st.warning("Failed to save search history")
                print("Failed to save search history")
            else:
                print("Successfully saved search history")
        except Exception as e:
            st.warning(f"Error saving search history: {str(e)}")
            print(f"Error saving search history: {str(e)}")
    
    # Cache the result
    st.session_state['response_cache'][cache_key] = (answer, contexts)
    return answer, contexts

# Add nullcontext for CPU mode
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *excinfo):
        pass

# Display chat messages
def display_chat_history():
    for message in st.session_state['messages']:
        if message['role'] == 'user':
            with st.container():
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="avatar">👤</div>
                    <div class="message">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="avatar">🤖</div>
                    <div class="message">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # If context is available, show it in an expander
                if 'context' in message and message['context']:
                    with st.expander("View retrieved context"):
                        for i, ctx in enumerate(message['context']):
                            st.markdown(f"**Reference {i+1}** (Similarity: {ctx['similarity']:.4f})")
                            st.markdown(f"**Q:** {ctx['question']}")
                            st.markdown(f"**A:** {ctx['answer']}")
                            st.markdown("---")

# Callback function for the chat input
def process_input():
    st.session_state.process_input = True

# Main application logic
def main():
    # Check if user is logged in
    if not st.session_state.is_logged_in:
        st.warning("Please log in to use the chatbot")
        return
    
    # Automatically load the model if not already loaded
    if not st.session_state['model_loaded']:
        try:
            # Load the embedding model if not already loaded
            if st.session_state['embedding_model'] is None:
                with st.spinner("Loading embedding model..."):
                    st.session_state['embedding_model'] = load_embedding_model()
                st.success("✅ Embedding model loaded!")
            
            # Load dataset
            data = load_dataset(file_path=data_path)
            
            if data is not None:
                # Prepare QA data
                st.session_state['questions'], st.session_state['answers'] = prepare_qa_data(data)
                
                # Create embeddings
                st.session_state['question_embeddings'] = create_embeddings(
                    st.session_state['questions'], 
                    st.session_state['embedding_model']
                )
                
                # Load the Granite model
                st.session_state['granite_model'], st.session_state['tokenizer'] = load_granite_model(
                    use_smaller_model=use_smaller_model,
                    force_cpu=force_cpu
                )
                
                if st.session_state['granite_model'] is not None:
                    # Mark model as loaded
                    st.session_state['model_loaded'] = True
                    st.success("✅ RAG system fully loaded and ready!")
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error setting up the RAG system: {str(e)}")
            st.info("💡 Tip: Use the 'Force CPU mode' option in the sidebar for more stable loading if you have GPU issues.")
            return

    # Display chat interface if the model is loaded
    st.subheader("Chat")
    
    # Show info box
    with st.container():
        st.markdown("""
        <div class="info-box">
            <h3>How to use this chatbot:</h3>
            <ul>
                <li>Type your question in the input box below</li>
                <li>The system will search through your knowledge base for relevant context</li>
                <li>The IBM Granite model will generate a response based on that context</li>
                <li>Your search history will be saved and used to improve future responses</li>
            </ul>
            <p>You can view the retrieved context by expanding the "View retrieved context" section below each response.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display previous messages
    display_chat_history()
    
    # Input for new message - Use a form to handle the input properly
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question:", key="user_input_field", placeholder="Type your question here...")
        submit_button = st.form_submit_button("Send", on_click=process_input)
    
    # Process the input after form submission
    if st.session_state.process_input and user_input:
        # Add user message to chat
        st.session_state['messages'].append({"role": "user", "content": user_input})
        
        # Generate bot response
        with st.spinner("Thinking..."):
            start_time = time.time()
            response, contexts = generate_rag_response(
                user_input, 
                top_k=top_k, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature
            )
            end_time = time.time()
        
        # Add bot response to chat
        st.session_state['messages'].append({
            "role": "assistant", 
            "content": response + f"\n\n<small>Response time: {end_time - start_time:.2f}s</small>", 
            "context": contexts
        })
        
        # Reset the process flag
        st.session_state.process_input = False
        
        # Force a rerun to update the UI
        st.rerun()

if __name__ == "__main__":
    main()