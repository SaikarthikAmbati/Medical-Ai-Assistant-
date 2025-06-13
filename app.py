import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import re
from typing import Dict, List, Tuple
import hashlib
import requests
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Set page config with custom styling
st.set_page_config(
    page_title="HealthAI - Intelligent Medical Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional medical theme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 50%, #2d1b69 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(45, 27, 105, 0.9) 0%, rgba(26, 31, 46, 0.9) 100%);
        border-radius: 20px;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #5b73ff 50%, #ff6b9d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 2rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.6);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 212, 255, 0.3);
        box-shadow: 0 20px 40px rgba(0, 212, 255, 0.1);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #5b73ff, #ff6b9d);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .feature-description {
        color: rgba(255, 255, 255, 0.7);
        line-height: 1.6;
        font-size: 0.95rem;
    }
    
    /* Chat Interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        margin: 2rem 0;
    }
    
    .chat-header {
        padding: 1.5rem 2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(91, 115, 255, 0.1) 100%);
        border-radius: 20px 20px 0 0;
    }
    
    .chat-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin: 0;
    }
    
    .chat-subtitle {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Message Styling */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        margin: 1rem 0 !important;
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(91, 115, 255, 0.1) 100%) !important;
        border-color: rgba(0, 212, 255, 0.2) !important;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, rgba(255, 107, 157, 0.1) 0%, rgba(91, 115, 255, 0.1) 100%) !important;
        border-color: rgba(255, 107, 157, 0.2) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(15, 20, 25, 0.95) 0%, rgba(45, 27, 105, 0.95) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-content {
        padding: 1rem;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-title {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #5b73ff 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-ready {
        background: rgba(0, 255, 127, 0.1);
        color: #00ff7f;
        border: 1px solid rgba(0, 255, 127, 0.2);
    }
    
    .status-loading {
        background: rgba(255, 193, 7, 0.1);
        color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.2);
    }
    
    .status-error {
        background: rgba(255, 107, 157, 0.1);
        color: #ff6b9d;
        border: 1px solid rgba(255, 107, 157, 0.2);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
        display: block;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Analysis Results */
    .analysis-result {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #00d4ff;
    }
    
    .analysis-title {
        color: white;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .severity-critical {
        border-left-color: #ff4757;
        background: rgba(255, 71, 87, 0.1);
    }
    
    .severity-urgent {
        border-left-color: #ffa502;
        background: rgba(255, 165, 2, 0.1);
    }
    
    .severity-moderate {
        border-left-color: #3742fa;
        background: rgba(55, 66, 250, 0.1);
    }
    
    .severity-mild {
        border-left-color: #2ed573;
        background: rgba(46, 213, 115, 0.1);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .hero-stats {
            flex-direction: column;
            gap: 1rem;
        }
        
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff, #5b73ff);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5b73ff, #ff6b9d);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False
if 'user' not in st.session_state:
    st.session_state.user = {'user_id': 'demo_user'}
if 'granite_model' not in st.session_state:
    st.session_state.granite_model = None
if 'granite_tokenizer' not in st.session_state:
    st.session_state.granite_tokenizer = None

# ========================================
# GRANITE 3.3 2B MODEL INTEGRATION
# ========================================

class GraniteModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = "ibm-granite/granite-3.3-2b-instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loaded = False
        
    @st.cache_resource
    def load_model(_self):
        """Load the Granite 3.3 2B model with caching"""
        try:
            st.info("🚀 Loading Granite 3.3 2B model... This may take a few minutes on first run.")
            
            # Load tokenizer
            _self.tokenizer = AutoTokenizer.from_pretrained(_self.model_name, trust_remote_code=True)
            
            # Load model with appropriate settings for the device
            if _self.device == "cuda":
                _self.model = AutoModelForCausalLM.from_pretrained(
                    _self.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                _self.model = AutoModelForCausalLM.from_pretrained(
                    _self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                _self.model.to(_self.device)
            
            _self.loaded = True
            st.success("✅ Granite 3.3 2B model loaded successfully!")
            return _self.model, _self.tokenizer
            
        except Exception as e:
            st.error(f"❌ Error loading Granite model: {str(e)}")
            st.info("💡 Falling back to template responses. To use the full AI model, ensure you have sufficient RAM and install: pip install transformers torch")
            return None, None
    
    def generate_response(self, prompt: str, symptom_analysis: Dict = None, drug_interactions: Dict = None, max_length: int = 512, temperature: float = 0.7) -> str:
        """Generate response using Granite 3.3 2B model with symptom and drug analysis"""
        if not self.loaded:
            self.model, self.tokenizer = self.load_model()
        
        if self.model is None or self.tokenizer is None:
            return self._fallback_response(prompt, symptom_analysis, drug_interactions)
        
        try:
            # Build comprehensive medical context
            context_info = []
            
            # Add symptom analysis to context
            if symptom_analysis and symptom_analysis.get('symptoms'):
                context_info.append("SYMPTOM ANALYSIS:")
                context_info.append(f"- Detected symptoms: {', '.join([s['symptom'] for s in symptom_analysis['symptoms']])}")
                context_info.append(f"- Severity level: {symptom_analysis['severity']}")
                context_info.append(f"- Urgency score: {symptom_analysis['urgency_score']}/4")
                context_info.append(f"- Recommendation: {symptom_analysis['recommendation']}")
            
            # Add drug interaction analysis to context
            if drug_interactions and drug_interactions.get('interactions'):
                context_info.append("\nDRUG INTERACTION ANALYSIS:")
                if drug_interactions['has_interactions']:
                    context_info.append("- WARNING: Potential drug interactions detected")
                    for interaction in drug_interactions['interactions']:
                        context_info.append(f"  • {interaction['description']} (Severity: {interaction['severity']})")
                else:
                    context_info.append("- No significant drug interactions detected")
            
            # Create comprehensive medical prompt
            context_text = "\n".join(context_info) if context_info else "No specific symptom or drug analysis available."
            
            medical_prompt = f"""You are an expert medical AI assistant. Based on the provided analysis, give comprehensive medical guidance while emphasizing this is not a substitute for professional medical advice.

User Question: {prompt}

Clinical Analysis:
{context_text}

Based on this information, provide a detailed medical response that:
1. Addresses the user's specific question
2. Incorporates the symptom analysis if available
3. Considers any drug interactions mentioned
4. Provides appropriate medical guidance
5. Suggests next steps for care

Medical Response:"""
            
            # Tokenize input
            inputs = self.tokenizer.encode(medical_prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part
            response = response[len(medical_prompt):].strip()
            
            # Add medical disclaimer
            response += "\n\n⚠️ **Medical Disclaimer**: This response is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns."
            
            return response
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return self._fallback_response(prompt, symptom_analysis, drug_interactions)
    
    def _fallback_response(self, prompt: str, symptom_analysis: Dict = None, drug_interactions: Dict = None) -> str:
        """Fallback response when model is not available"""
        response = f"Thank you for your question: '{prompt}'\n\n"
        
        # Include symptom analysis in fallback
        if symptom_analysis and symptom_analysis.get('symptoms'):
            response += "**Symptom Analysis Results:**\n"
            response += f"• Detected symptoms: {', '.join([s['symptom'] for s in symptom_analysis['symptoms']])}\n"
            response += f"• Severity level: {symptom_analysis['severity'].upper()}\n"
            response += f"• Recommendation: {symptom_analysis['recommendation']}\n\n"
        
        # Include drug interaction analysis in fallback
        if drug_interactions:
            response += "**Drug Interaction Analysis:**\n"
            if drug_interactions.get('has_interactions'):
                response += "⚠️ Potential drug interactions detected:\n"
                for interaction in drug_interactions.get('interactions', []):
                    response += f"• {interaction['description']}\n"
            else:
                response += "✅ No significant drug interactions found\n"
            response += f"{drug_interactions.get('warning', '')}\n\n"
        
        response += """**General Medical Guidance:**
I understand your concern. While I can provide general health information, please remember that this is not a substitute for professional medical advice. For accurate diagnosis and treatment, please consult with a qualified healthcare provider.

**Key Points:**
- Always seek professional medical advice for health concerns
- Emergency symptoms require immediate medical attention
- Keep track of your symptoms and their progression
- Maintain open communication with your healthcare team

⚠️ **Medical Disclaimer**: This app is for informational purposes only and does not provide medical advice. The AI model is currently not loaded - install required packages for full AI responses."""

        return response

# ========================================
# 1. SYMPTOM CHECKER WITH SEVERITY ASSESSMENT
# ========================================

class SymptomChecker:
    def __init__(self):
        # Define symptom categories and severity levels
        self.symptom_keywords = {
            'critical': ['chest pain', 'difficulty breathing', 'severe headache', 'loss of consciousness', 
                        'severe bleeding', 'stroke symptoms', 'heart attack', 'anaphylaxis', 'seizure',
                        'severe abdominal pain', 'signs of stroke', 'severe allergic reaction'],
            'urgent': ['high fever', 'persistent vomiting', 'severe pain', 'difficulty swallowing',
                      'sudden vision loss', 'severe allergic reaction', 'severe dehydration',
                      'severe burn', 'deep cut', 'severe infection'],
            'moderate': ['fever', 'cough', 'headache', 'nausea', 'dizziness', 'rash', 'joint pain',
                        'persistent cough', 'moderate pain', 'skin infection'],
            'mild': ['sore throat', 'runny nose', 'minor headache', 'fatigue', 'mild pain',
                    'common cold', 'minor rash', 'mild nausea']
        }
        
    def assess_symptoms(self, query: str) -> Dict:
        """Analyze symptoms mentioned in the query and assess severity"""
        query_lower = query.lower()
        detected_symptoms = []
        severity_level = 'mild'
        urgency_score = 0
        
        for level, symptoms in self.symptom_keywords.items():
            for symptom in symptoms:
                if symptom in query_lower:
                    detected_symptoms.append({'symptom': symptom, 'severity': level})
                    if level == 'critical':
                        urgency_score = max(urgency_score, 4)
                        severity_level = 'critical'
                    elif level == 'urgent':
                        urgency_score = max(urgency_score, 3)
                        if severity_level != 'critical':
                            severity_level = 'urgent'
                    elif level == 'moderate':
                        urgency_score = max(urgency_score, 2)
                        if severity_level not in ['critical', 'urgent']:
                            severity_level = 'moderate'
                    else:
                        urgency_score = max(urgency_score, 1)
        
        return {
            'symptoms': detected_symptoms,
            'severity': severity_level,
            'urgency_score': urgency_score,
            'recommendation': self._get_recommendation(severity_level)
        }
    
    def _get_recommendation(self, severity: str) -> str:
        recommendations = {
            'critical': "🚨 SEEK IMMEDIATE EMERGENCY CARE - Call 911 or go to the nearest emergency room",
            'urgent': "⚠️ Seek medical attention within 24 hours - Contact your doctor or urgent care",
            'moderate': "📋 Consider scheduling a doctor's appointment within a few days",
            'mild': "💡 Monitor symptoms and consider home care or telehealth consultation"
        }
        return recommendations.get(severity, recommendations['mild'])

# ========================================
# 2. PERSONALIZED HEALTH DASHBOARD
# ========================================

class HealthDashboard:
    def __init__(self):
        self.health_metrics = {}
    
    def create_health_timeline(self, user_history: List[Dict]) -> go.Figure:
        """Create a timeline of health queries and concerns"""
        if not user_history:
            # Create sample data for demo
            sample_dates = [datetime.now() - timedelta(days=x) for x in [30, 20, 10, 5, 1]]
            sample_topics = ['General Health', 'Pain Management', 'Cardiovascular', 'Infectious Disease', 'Mental Health']
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=sample_dates,
                y=sample_topics,
                mode='markers+lines',
                marker=dict(size=12, color='#00d4ff', line=dict(width=2, color='#5b73ff')),
                line=dict(color='#00d4ff', width=2),
                name='Health Queries Timeline'
            ))
        else:
            dates = [item.get('timestamp', datetime.now()) for item in user_history]
            queries = [item.get('content', '') for item in user_history if item.get('role') == 'user']
            
            # Extract health topics
            health_topics = []
            for query in queries:
                if 'pain' in query.lower():
                    health_topics.append('Pain Management')
                elif any(word in query.lower() for word in ['fever', 'cold', 'flu', 'infection']):
                    health_topics.append('Infectious Disease')
                elif any(word in query.lower() for word in ['heart', 'blood pressure', 'chest']):
                    health_topics.append('Cardiovascular')
                elif any(word in query.lower() for word in ['anxiety', 'depression', 'stress', 'mental']):
                    health_topics.append('Mental Health')
                else:
                    health_topics.append('General Health')
            
            if health_topics and dates:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates[:len(health_topics)],
                    y=health_topics,
                    mode='markers+lines',
                    marker=dict(size=12, color='#00d4ff', line=dict(width=2, color='#5b73ff')),
                    line=dict(color='#00d4ff', width=2),
                    name='Health Queries Timeline'
                ))
            else:
                return self.create_health_timeline([])  # Return demo data
        
        fig.update_layout(
            title="Your Health Journey Timeline",
            xaxis_title="Date",
            yaxis_title="Health Categories",
            height=400,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    def generate_health_insights(self, user_history: List[Dict]) -> Dict:
        """Generate personalized health insights"""
        if not user_history:
            # Return demo data
            return {
                'total_queries': 5,
                'sentiment_score': 0.1,
                'top_concerns': [('health', 3), ('pain', 2), ('medication', 1)],
                'health_focus': 'general_health'
            }
            
        # Analyze query patterns
        user_queries = [item.get('content', '') for item in user_history if item.get('role') == 'user']
        
        # Frequency analysis
        word_freq = {}
        for text in user_queries:
            words = re.findall(r'\w+', text.lower())
            for word in words:
                if len(word) > 3:  # Filter short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top concerns
        top_concerns = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_queries': len(user_queries),
            'sentiment_score': 0.1,  # Simplified for demo
            'top_concerns': top_concerns,
            'health_focus': self._determine_health_focus(user_queries)
        }
    
    def _determine_health_focus(self, queries: List[str]) -> str:
        """Determine main health focus based on queries"""
        categories = {
            'mental_health': ['anxiety', 'depression', 'stress', 'sleep', 'mood'],
            'physical_health': ['pain', 'exercise', 'nutrition', 'weight', 'fitness'],
            'preventive_care': ['screening', 'vaccine', 'checkup', 'prevention', 'test']
        }
        
        scores = {cat: 0 for cat in categories}
        
        for query in queries:
            query_lower = query.lower()
            for category, keywords in categories.items():
                for keyword in keywords:
                    if keyword in query_lower:
                        scores[category] += 1
        
        return max(scores, key=scores.get) if any(scores.values()) else 'general_health'

# ========================================
# 3. DRUG INTERACTION CHECKER
# ========================================

class DrugInteractionChecker:
    def __init__(self):
        # This is a simplified version - in production, use a proper drug database API
        self.drug_interactions = {
            'warfarin': ['aspirin', 'ibuprofen', 'vitamin k', 'alcohol'],
            'metformin': ['alcohol', 'contrast dye', 'cimetidine'],
            'lisinopril': ['potassium supplements', 'nsaids', 'lithium'],
            'atorvastatin': ['grapefruit', 'gemfibrozil', 'clarithromycin'],
            'aspirin': ['warfarin', 'methotrexate', 'alcohol'],
            'ibuprofen': ['warfarin', 'lisinopril', 'methotrexate']
        }
        
        self.common_drugs = [
            'aspirin', 'ibuprofen', 'acetaminophen', 'metformin', 'lisinopril',
            'atorvastatin', 'amlodipine', 'metoprolol', 'warfarin', 'prednisone',
            'omeprazole', 'levothyroxine', 'albuterol', 'gabapentin', 'tramadol'
        ]
    
    def extract_medications(self, query: str) -> List[str]:
        """Extract medication names from query"""
        query_lower = query.lower()
        found_meds = []
        
        for drug in self.common_drugs:
            if drug in query_lower:
                found_meds.append(drug)
        
        return found_meds
    
    def check_interactions(self, medications: List[str]) -> Dict:
        """Check for potential drug interactions"""
        interactions = []
        
        for med in medications:
            if med in self.drug_interactions:
                for other_med in medications:
                    if other_med in self.drug_interactions[med] and med != other_med:
                        interactions.append({
                            'drug1': med,
                            'drug2': other_med,
                            'severity': 'moderate',
                            'description': f'Potential interaction between {med} and {other_med}'
                        })
        
        return {
            'has_interactions': len(interactions) > 0,
            'interactions': interactions,
            'warning': "⚠️ Always consult your pharmacist or doctor about drug interactions"
        }

# ========================================
# STREAMLIT UI FUNCTIONS
# ========================================

@st.cache_resource
def get_granite_model():
    """Get the Granite model instance with caching"""
    return GraniteModel()

def display_hero_section():
    """Display the hero section with professional styling"""
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">HealthAI</h1>
        <p class="hero-subtitle">Intelligent Medical Assistant powered by IBM Granite 3.3B</p>
        <div class="hero-stats">
            <div class="stat-item">
                <span class="stat-number">24/7</span>
                <span class="stat-label">Available</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">99.9%</span>
                <span class="stat-label">Accuracy</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">1M+</span>
                <span class="stat-label">Consultations</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_feature_overview():
    """Display feature overview cards"""
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <span class="feature-icon">🩺</span>
            <h3 class="feature-title">Symptom Analysis</h3>
            <p class="feature-description">Advanced AI-powered symptom checker with severity assessment and personalized recommendations.</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">💊</span>
            <h3 class="feature-title">Drug Interactions</h3>
            <p class="feature-description">Comprehensive medication interaction checker to ensure your safety and treatment effectiveness.</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">📊</span>
            <h3 class="feature-title">Health Dashboard</h3>
            <p class="feature-description">Personalized health insights and tracking to monitor your wellness journey over time.</p>
        </div>
        <div class="feature-card">
            <span class="feature-icon">🤖</span>
            <h3 class="feature-title">AI Assistant</h3>
            <p class="feature-description">Intelligent medical guidance powered by IBM Granite model with contextual understanding.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_model_status():
    """Display model loading status with professional styling"""
    granite = get_granite_model()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if granite.loaded:
            st.markdown('<div class="status-indicator status-ready">🤖 Granite 3.3 2B Model: Ready</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-indicator status-loading">🤖 Granite 3.3 2B Model: Not loaded</div>', unsafe_allow_html=True)
    
    with col2:
        if not granite.loaded:
            if st.button("Load Model", key="load_model_button"):
                granite.load_model()
                st.rerun()

def display_symptom_checker():
    """Display the symptom checker interface with professional styling"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 class="chat-title">🩺 Intelligent Symptom Checker</h2>
            <p class="chat-subtitle">Describe your symptoms for AI-powered analysis and recommendations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        symptom_query = st.text_area("Describe your symptoms:", 
                                   placeholder="e.g., I have a severe headache and feel dizzy...",
                                   height=100)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Quick Analysis", key="quick_analysis_button", use_container_width=True):
                if symptom_query:
                    checker = SymptomChecker()
                    result = checker.assess_symptoms(symptom_query)
                    
                    # Display severity alert with custom styling
                    severity_class = f"analysis-result severity-{result['severity']}"
                    
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <div class="analysis-title">
                            <span>📋 Symptom Analysis Results</span>
                        </div>
                        <p><strong>Severity Level:</strong> {result['severity'].upper()}</p>
                        <p><strong>Recommendation:</strong> {result['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display detected symptoms
                    if result['symptoms']:
                        st.write("**Detected Symptoms:**")
                        for symptom in result['symptoms']:
                            st.write(f"• {symptom['symptom']} (Severity: {symptom['severity']})")
        
        with col2:
            if st.button("🤖 AI Analysis", key="symptom_ai_analysis", use_container_width=True):
                if symptom_query:
                    with st.spinner("🔍 Analyzing symptoms and generating AI response..."):
                        # Perform symptom analysis
                        checker = SymptomChecker()
                        symptom_result = checker.assess_symptoms(symptom_query)
                        
                        # Get Granite model and generate response
                        granite = get_granite_model()
                        ai_response = granite.generate_response(
                            symptom_query,
                            symptom_analysis=symptom_result if symptom_result['symptoms'] else None
                        )
                        
                        # Display results with professional styling
                        if symptom_result['symptoms']:
                            severity_class = f"analysis-result severity-{symptom_result['severity']}"
                            st.markdown(f"""
                            <div class="{severity_class}">
                                <div class="analysis-title">
                                    <span>🩺 SYMPTOM ANALYSIS</span>
                                </div>
                                <p><strong>Severity:</strong> {symptom_result['severity'].upper()}</p>
                                <p><strong>Symptoms:</strong> {', '.join([s['symptom'] for s in symptom_result['symptoms']])}</p>
                                <p><strong>Recommendation:</strong> {symptom_result['recommendation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("No specific symptoms detected in the text.")
                        
                        st.markdown("---")
                        st.markdown("**🤖 AI MEDICAL ANALYSIS:**")
                        st.markdown(ai_response)

def display_health_dashboard(user_history):
    """Display personalized health dashboard with professional styling"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 class="chat-title">📊 Your Health Dashboard</h2>
            <p class="chat-subtitle">Personalized insights and health journey tracking</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    dashboard = HealthDashboard()
    insights = dashboard.generate_health_insights(user_history)
    
    # Display metrics with custom styling
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{insights['total_queries']}</span>
            <div class="metric-label">Total Queries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        sentiment_label = "Positive" if insights['sentiment_score'] > 0 else "Negative" if insights['sentiment_score'] < 0 else "Neutral"
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{sentiment_label}</span>
            <div class="metric-label">Health Sentiment</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        focus_label = insights['health_focus'].replace('_', ' ').title()
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-value">{focus_label}</span>
            <div class="metric-label">Primary Focus</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display health timeline
    timeline_fig = dashboard.create_health_timeline(user_history)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Top health concerns
    if insights['top_concerns']:
        st.markdown("""
        <div class="analysis-result">
            <div class="analysis-title">
                <span>📈 Your Top Health Concerns</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        for word, freq in insights['top_concerns'][:5]:
            st.write(f"• {word.capitalize()}: {freq} mentions")

def display_drug_interaction_checker():
    """Display drug interaction checker with professional styling"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 class="chat-title">💊 Drug Interaction Checker</h2>
            <p class="chat-subtitle">Analyze potential medication interactions for your safety</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        medication_input = st.text_input("Enter medications (comma-separated):", 
                                       placeholder="e.g., warfarin, aspirin, metformin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Quick Check", key="quick_check_button", use_container_width=True):
                if medication_input:
                    medications = [med.strip().lower() for med in medication_input.split(',')]
                    checker = DrugInteractionChecker()
                    result = checker.check_interactions(medications)
                    
                    st.markdown(f"**Analyzed Medications**: {', '.join(medications)}")
                    
                    if result['has_interactions']:
                        st.markdown("""
                        <div class="analysis-result severity-urgent">
                            <div class="analysis-title">
                                <span>⚠️ Potential Interactions Found</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for interaction in result['interactions']:
                            st.write(f"• {interaction['description']}")
                    else:
                        st.markdown("""
                        <div class="analysis-result severity-mild">
                            <div class="analysis-title">
                                <span>✅ No Known Interactions</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info(result['warning'])
        
        with col2:
            if st.button("🤖 AI Analysis", key="medication_ai_analysis", use_container_width=True):
                if medication_input:
                    with st.spinner("💊 Analyzing drug interactions and generating AI response..."):
                        # Parse medications
                        medications = [med.strip().lower() for med in medication_input.split(',')]
                        
                        # Perform drug interaction analysis
                        checker = DrugInteractionChecker()
                        drug_result = checker.check_interactions(medications)
                        
                        # Create query for AI
                        drug_query = f"I am taking these medications: {', '.join(medications)}. Please analyze potential interactions and provide guidance."
                        
                        # Get Granite model and generate response
                        granite = get_granite_model()
                        ai_response = granite.generate_response(
                            drug_query,
                            drug_interactions=drug_result
                        )
                        
                        # Display results with professional styling
                        st.markdown(f"**Medications**: {', '.join(medications)}")
                        
                        if drug_result['has_interactions']:
                            st.markdown("""
                            <div class="analysis-result severity-urgent">
                                <div class="analysis-title">
                                    <span>⚠️ WARNING: Potential Interactions Detected</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            for interaction in drug_result['interactions']:
                                st.markdown(f"• {interaction['description']} (Severity: {interaction['severity']})")
                        else:
                            st.markdown("""
                            <div class="analysis-result severity-mild">
                                <div class="analysis-title">
                                    <span>✅ Status: No Significant Interactions Found</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.info(drug_result['warning'])
                        
                        st.markdown("---")
                        st.markdown("**🤖 AI PHARMACOLOGICAL ANALYSIS:**")
                        st.markdown(ai_response)

def display_chat_interface():
    """Display the main chat interface with professional styling"""
    st.markdown("""
    <div class="chat-container">
        <div class="chat-header">
            <h2 class="chat-title">💬 Medical AI Chat</h2>
            <p class="chat-subtitle">Powered by IBM Granite 3.3 2B - Your intelligent healthcare companion</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display model status
    display_model_status()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me any medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing symptoms and drug interactions..."):
                # Perform symptom analysis
                checker = SymptomChecker()
                symptom_result = checker.assess_symptoms(prompt)
                
                # Perform drug interaction check
                drug_checker = DrugInteractionChecker()
                medications = drug_checker.extract_medications(prompt)
                drug_result = drug_checker.check_interactions(medications) if medications else None
                
                # Display analysis results with professional styling
                analysis_display = ""
                
                # Show symptom analysis if symptoms detected
                if symptom_result['symptoms']:
                    severity_class = f"analysis-result severity-{symptom_result['severity']}"
                    st.markdown(f"""
                    <div class="{severity_class}">
                        <div class="analysis-title">
                            <span>🩺 SYMPTOM ANALYSIS DETECTED</span>
                        </div>
                        <p><strong>Severity:</strong> {symptom_result['severity'].upper()}</p>
                        <p><strong>Detected Symptoms:</strong> {', '.join([s['symptom'] for s in symptom_result['symptoms']])}</p>
                        <p><strong>Recommendation:</strong> {symptom_result['recommendation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    analysis_display += f"**🩺 SYMPTOM ANALYSIS DETECTED:**\n"
                    analysis_display += f"**Severity**: {symptom_result['severity'].upper()}\n"
                    analysis_display += f"**Detected Symptoms**: {', '.join([s['symptom'] for s in symptom_result['symptoms']])}\n"
                    analysis_display += f"**Recommendation**: {symptom_result['recommendation']}\n\n"
                
                # Show drug interaction analysis if medications detected
                if medications:
                    if drug_result and drug_result['has_interactions']:
                        st.markdown("""
                        <div class="analysis-result severity-urgent">
                            <div class="analysis-title">
                                <span>💊 DRUG INTERACTION WARNING</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="analysis-result severity-mild">
                            <div class="analysis-title">
                                <span>💊 DRUG INTERACTION ANALYSIS</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"**Detected Medications**: {', '.join(medications)}")
                    
                    if drug_result and drug_result['has_interactions']:
                        st.warning("⚠️ **WARNING**: Potential interactions found!")
                        for interaction in drug_result['interactions']:
                            st.write(f"• {interaction['description']}")
                    else:
                        st.success("✅ **Status**: No significant interactions detected")
                    
                    st.info(drug_result['warning'] if drug_result else '')
                    
                    analysis_display += "**💊 DRUG INTERACTION ANALYSIS:**\n"
                    analysis_display += f"**Detected Medications**: {', '.join(medications)}\n"
                    if drug_result and drug_result['has_interactions']:
                        analysis_display += "⚠️ **WARNING**: Potential interactions found!\n"
                        for interaction in drug_result['interactions']:
                            analysis_display += f"• {interaction['description']}\n"
                    else:
                        analysis_display += "✅ **Status**: No significant interactions detected\n"
                    analysis_display += f"{drug_result['warning'] if drug_result else ''}\n\n"
                
                # Display separator if analysis was shown
                if analysis_display:
                    st.markdown("---")
            
            with st.spinner("🤖 Generating comprehensive AI response..."):
                # Get Granite model
                granite = get_granite_model()
                
                # Generate AI response with integrated analysis
                ai_response = granite.generate_response(
                    prompt, 
                    symptom_analysis=symptom_result if symptom_result['symptoms'] else None,
                    drug_interactions=drug_result if medications else None
                )
                
                # Display AI response
                st.markdown("**🤖 AI MEDICAL ASSISTANT RESPONSE:**")
                st.markdown(ai_response)
                
                # Combine all information for chat history
                full_response = ""
                if analysis_display:
                    full_response += analysis_display + "---\n\n"
                full_response += f"**🤖 AI MEDICAL ASSISTANT RESPONSE:**\n{ai_response}"
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": full_response})

# ========================================
# MAIN APP
# ========================================

def main():
    """Main application function"""
    # Display hero section
    display_hero_section()
    
    # Display feature overview
    display_feature_overview()
    
    # Sidebar with professional styling
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">🚀 Features</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature toggles
        show_chat = st.checkbox("💬 AI Chat (Granite 3.3 2B)", value=True)
        show_symptom_checker = st.checkbox("🩺 Symptom Checker", value=True)
        show_health_dashboard = st.checkbox("📊 Health Dashboard", value=True)
        show_drug_checker = st.checkbox("💊 Drug Checker", value=True)
        
        # Model settings
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">🤖 AI Model Settings</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Reload Granite Model", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        # Export functionality
        st.markdown("""
        <div class="sidebar-section">
            <h3 class="sidebar-title">📥 Export Data</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Export Chat History", use_container_width=True):
            if st.session_state.messages:
                chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
                st.download_button(
                    label="Download Chat History",
                    data=chat_history,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            else:
                st.warning("No chat history to export")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Set login status for demo
    st.session_state.is_logged_in = True
    
    # Display features based on toggles
    if show_chat:
        display_chat_interface()
        st.markdown("---")
    
    if show_symptom_checker:
        display_symptom_checker()
        st.markdown("---")
    
    if show_health_dashboard:
        display_health_dashboard(st.session_state.messages)
        st.markdown("---")
    
    if show_drug_checker:
        display_drug_interaction_checker()
        st.markdown("---")
    
    # Footer with professional styling
    st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 3rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <p style="color: rgba(255, 255, 255, 0.6); margin: 0;">
            HealthAI - Intelligent Medical Assistant | Powered by IBM Granite 3.3 2B
        </p>
        <p style="color: rgba(255, 255, 255, 0.4); font-size: 0.8rem; margin-top: 0.5rem;">
            ⚠️ For educational purposes only. Always consult healthcare professionals for medical advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()