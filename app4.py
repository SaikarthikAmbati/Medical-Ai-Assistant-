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

# Set page config
st.set_page_config(
    page_title="Enhanced Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            st.info("Loading Granite 3.3 2B model... This may take a few minutes on first run.")
            
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
                marker=dict(size=12, color='lightblue', line=dict(width=2, color='darkblue')),
                line=dict(color='lightblue', width=2),
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
                    marker=dict(size=12, color='lightblue', line=dict(width=2, color='darkblue')),
                    line=dict(color='lightblue', width=2),
                    name='Health Queries Timeline'
                ))
            else:
                return self.create_health_timeline([])  # Return demo data
        
        fig.update_layout(
            title="Your Health Journey Timeline",
            xaxis_title="Date",
            yaxis_title="Health Categories",
            height=400,
            template="plotly_white"
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
# 4. VOICE INTERACTION CAPABILITY
# ========================================

class VoiceInterface:
    def __init__(self):
        self.voice_enabled = False
        self.recognizer = None
        self.sr = None
    
    def setup_voice_recognition(self):
        """Setup voice recognition capabilities"""
        try:
            import speech_recognition as sr
            self.sr = sr
            self.recognizer = sr.Recognizer()
            self.voice_enabled = True
            return True
        except ImportError:
            st.warning("Voice recognition not available. Install speech_recognition package: pip install SpeechRecognition")
            return False
    
    def process_voice_input(self) -> str:
        """Process voice input and convert to text"""
        if not self.voice_enabled or not self.sr:
            return ""
        
        try:
            with self.sr.Microphone() as source:
                st.info("🎤 Listening... Please speak your question.")
                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)
                return text
        except Exception as e:
            st.error(f"Voice recognition error: {str(e)}")
            return ""

# ========================================
# STREAMLIT UI FUNCTIONS
# ========================================

@st.cache_resource
def get_granite_model():
    """Get the Granite model instance with caching"""
    return GraniteModel()

def display_model_status():
    """Display model loading status"""
    granite = get_granite_model()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if granite.loaded:
            st.success("🤖 Granite 3.3 2B Model: Ready")
        else:
            st.warning("🤖 Granite 3.3 2B Model: Not loaded")
    
    with col2:
        if not granite.loaded:
            if st.button("Load Model", key="load_model_button"):
                granite.load_model()
                st.rerun()

def display_symptom_checker():
    """Display the symptom checker interface with AI integration"""
    st.subheader("🩺 Intelligent Symptom Checker")
    
    with st.expander("Analyze Your Symptoms", expanded=False):
        symptom_query = st.text_area("Describe your symptoms:", 
                                   placeholder="e.g., I have a severe headache and feel dizzy...")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Quick Analysis", key="quick_analysis_button"):
                if symptom_query:
                    checker = SymptomChecker()
                    result = checker.assess_symptoms(symptom_query)
                    
                    # Display severity alert
                    if result['severity'] == 'critical':
                        st.error(result['recommendation'])
                    elif result['severity'] == 'urgent':
                        st.warning(result['recommendation'])
                    elif result['severity'] == 'moderate':
                        st.info(result['recommendation'])
                    else:
                        st.success(result['recommendation'])
                    
                    # Display detected symptoms
                    if result['symptoms']:
                        st.write("**Detected Symptoms:**")
                        for symptom in result['symptoms']:
                            st.write(f"• {symptom['symptom']} (Severity: {symptom['severity']})")
        
        with col2:
            if st.button("🤖 AI Analysis", key="symptom_ai_analysis"):
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
                        
                        # Display results
                        st.markdown("**🩺 SYMPTOM ANALYSIS:**")
                        if symptom_result['symptoms']:
                            severity_color = {
                                'critical': '🚨',
                                'urgent': '⚠️', 
                                'moderate': '📋',
                                'mild': '💡'
                            }
                            st.markdown(f"{severity_color.get(symptom_result['severity'], '💡')} **Severity**: {symptom_result['severity'].upper()}")
                            st.markdown(f"**Symptoms**: {', '.join([s['symptom'] for s in symptom_result['symptoms']])}")
                            st.markdown(f"**Recommendation**: {symptom_result['recommendation']}")
                        else:
                            st.info("No specific symptoms detected in the text.")
                        
                        st.markdown("---")
                        st.markdown("**🤖 AI MEDICAL ANALYSIS:**")
                        st.markdown(ai_response)
    
    return None

def display_health_dashboard(user_history):
    """Display personalized health dashboard"""
    st.subheader("📊 Your Health Dashboard")
    
    dashboard = HealthDashboard()
    insights = dashboard.generate_health_insights(user_history)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", insights['total_queries'])
    
    with col2:
        sentiment_label = "Positive" if insights['sentiment_score'] > 0 else "Negative" if insights['sentiment_score'] < 0 else "Neutral"
        st.metric("Health Sentiment", sentiment_label)
    
    with col3:
        st.metric("Primary Focus", insights['health_focus'].replace('_', ' ').title())
    
    # Display health timeline
    timeline_fig = dashboard.create_health_timeline(user_history)
    if timeline_fig:
        st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Top health concerns
    if insights['top_concerns']:
        st.subheader("Your Top Health Concerns")
        st.write("**Most Discussed Topics:**")
        for word, freq in insights['top_concerns'][:5]:
            st.write(f"• {word.capitalize()}: {freq} mentions")

def display_drug_interaction_checker():
    """Display drug interaction checker with AI integration"""
    st.subheader("💊 Drug Interaction Checker")
    
    with st.expander("Check Drug Interactions", expanded=False):
        medication_input = st.text_input("Enter medications (comma-separated):", 
                                       placeholder="e.g., warfarin, aspirin, metformin")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Quick Check", key="quick_check_button"):
                if medication_input:
                    medications = [med.strip().lower() for med in medication_input.split(',')]
                    checker = DrugInteractionChecker()
                    result = checker.check_interactions(medications)
                    
                    st.markdown(f"**Analyzed Medications**: {', '.join(medications)}")
                    
                    if result['has_interactions']:
                        st.warning("⚠️ Potential interactions found!")
                        for interaction in result['interactions']:
                            st.write(f"• {interaction['description']}")
                    else:
                        st.success("✅ No known interactions found")
                    
                    st.info(result['warning'])
        
        with col2:
            if st.button("🤖 AI Analysis", key="medication_ai_analysis"):
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
                        
                        # Display results
                        st.markdown("**💊 DRUG INTERACTION ANALYSIS:**")
                        st.markdown(f"**Medications**: {', '.join(medications)}")
                        
                        if drug_result['has_interactions']:
                            st.warning("⚠️ **WARNING**: Potential interactions detected!")
                            for interaction in drug_result['interactions']:
                                st.markdown(f"• {interaction['description']} (Severity: {interaction['severity']})")
                        else:
                            st.success("✅ **Status**: No significant interactions found")
                        
                        st.info(drug_result['warning'])
                        
                        st.markdown("---")
                        st.markdown("**🤖 AI PHARMACOLOGICAL ANALYSIS:**")
                        st.markdown(ai_response)

def display_voice_interface():
    """Display voice interface"""
    st.subheader("🎤 Voice Assistant")
    
    with st.expander("Voice Interaction", expanded=False):
        voice_interface = VoiceInterface()
        
        if voice_interface.setup_voice_recognition():
            if st.button("🎤 Start Voice Input", key="start_voice_input_button"):
                voice_text = voice_interface.process_voice_input()
                if voice_text:
                    st.success(f"You said: {voice_text}")
                    return voice_text
        else:
            st.info("Voice recognition not available. Please install required packages.")
    
    return None

def display_chat_interface():
    """Display the main chat interface with integrated analysis"""
    st.subheader("💬 Medical AI Chat (Powered by Granite 3.3 2B)")
    
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
                
                # Display analysis results
                analysis_display = ""
                
                # Show symptom analysis if symptoms detected
                if symptom_result['symptoms']:
                    analysis_display += "**🩺 SYMPTOM ANALYSIS DETECTED:**\n"
                    severity_color = {
                        'critical': '🚨',
                        'urgent': '⚠️',
                        'moderate': '📋',
                        'mild': '💡'
                    }
                    analysis_display += f"{severity_color.get(symptom_result['severity'], '💡')} **Severity**: {symptom_result['severity'].upper()}\n"
                    analysis_display += f"**Detected Symptoms**: {', '.join([s['symptom'] for s in symptom_result['symptoms']])}\n"
                    analysis_display += f"**Recommendation**: {symptom_result['recommendation']}\n\n"
                
                # Show drug interaction analysis if medications detected
                if medications:
                    analysis_display += "**💊 DRUG INTERACTION ANALYSIS:**\n"
                    analysis_display += f"**Detected Medications**: {', '.join(medications)}\n"
                    if drug_result and drug_result['has_interactions']:
                        analysis_display += "⚠️ **WARNING**: Potential interactions found!\n"
                        for interaction in drug_result['interactions']:
                            analysis_display += f"• {interaction['description']}\n"
                    else:
                        analysis_display += "✅ **Status**: No significant interactions detected\n"
                    analysis_display += f"{drug_result['warning'] if drug_result else ''}\n\n"
                
                # Display analysis if any found
                if analysis_display:
                    st.markdown(analysis_display)
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
    st.title("🩺 Enhanced Medical AI Assistant")
    st.markdown("🤖 **Powered by IBM Granite 3.3 2B** - Your intelligent healthcare companion with advanced AI")
    
    # Sidebar with feature toggles
    st.sidebar.title("🚀 Features")
    st.sidebar.markdown("---")
    
    # Feature toggles
    show_chat = st.sidebar.checkbox("💬 AI Chat (Granite 3.3 2B)", value=True)
    show_symptom_checker = st.sidebar.checkbox("🩺 Symptom Checker", value=True)
    show_health_dashboard = st.sidebar.checkbox("📊 Health Dashboard", value=True)
    show_drug_checker = st.sidebar.checkbox("💊 Drug Checker", value=True)
    show_voice_interface = st.sidebar.checkbox("🎤 Voice Assistant", value=False)
    
    # Model settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🤖 AI Model Settings**")
    
    if st.sidebar.button("🔄 Reload Granite Model"):
        st.cache_resource.clear()
        st.rerun()
    
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
    
    if show_voice_interface:
        voice_input = display_voice_interface()
        if voice_input:
            st.success(f"Voice input received: {voice_input}")
        st.markdown("---")
    
    # Footer
    
    # Export functionality
    if st.sidebar.button("📥 Export Chat History"):
        if st.session_state.messages:
            chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
            st.sidebar.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.sidebar.warning("No chat history to export")

if __name__ == "__main__":
    main()