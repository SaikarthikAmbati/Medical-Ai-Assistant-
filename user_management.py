import streamlit as st
import pandas as pd
import sqlite3
import hashlib
import json
from datetime import datetime
import os

class UserManagement:
    def __init__(self):
        # Use absolute path for database
        self.db_path = os.path.abspath("user_data.db")
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Create users table
            c.execute('''CREATE TABLE IF NOT EXISTS users
                        (user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                         username TEXT UNIQUE NOT NULL,
                         password_hash TEXT NOT NULL,
                         email TEXT UNIQUE NOT NULL,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
            
            # Create search_history table with proper foreign key constraint
            c.execute('''CREATE TABLE IF NOT EXISTS search_history
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         user_id INTEGER NOT NULL,
                         query TEXT NOT NULL,
                         response TEXT NOT NULL,
                         context TEXT,
                         timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                         FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE)''')
            
            # Create index for faster queries
            c.execute('''CREATE INDEX IF NOT EXISTS idx_search_history_user_id 
                        ON search_history(user_id)''')
            
            conn.commit()
            conn.close()
            print(f"Database initialized at: {self.db_path}")
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            raise
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username, password, email):
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if username or email already exists
            c.execute("SELECT * FROM users WHERE username = ? OR email = ?", (username, email))
            if c.fetchone():
                return False, "Username or email already exists"
            
            # Hash password and insert new user
            password_hash = self.hash_password(password)
            c.execute("INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)",
                     (username, password_hash, email))
            
            conn.commit()
            conn.close()
            return True, "Registration successful"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, username, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check credentials
            password_hash = self.hash_password(password)
            c.execute("SELECT user_id, username FROM users WHERE username = ? AND password_hash = ?",
                     (username, password_hash))
            user = c.fetchone()
            
            conn.close()
            
            if user:
                return True, {"user_id": user[0], "username": user[1]}
            return False, "Invalid username or password"
        except Exception as e:
            return False, f"Login failed: {str(e)}"
    
    def save_search_history(self, user_id, query, response, context):
        """Save user's search history"""
        try:
            # Verify user exists before saving
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not c.fetchone():
                print(f"User {user_id} not found")
                return False
            
            # Convert context to JSON string
            context_json = json.dumps(context)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Debug print
            print(f"Saving history for user {user_id}:")
            print(f"Query: {query}")
            print(f"Response: {response[:100]}...")
            print(f"Timestamp: {timestamp}")
            
            # Insert with explicit timestamp
            c.execute("""INSERT INTO search_history 
                        (user_id, query, response, context, timestamp) 
                        VALUES (?, ?, ?, ?, ?)""",
                     (user_id, query, response, context_json, timestamp))
            
            conn.commit()
            conn.close()
            print("History saved successfully")
            return True
        except Exception as e:
            print(f"Error saving search history: {str(e)}")
            return False
    
    def get_user_history(self, user_id, limit=10):
        """Retrieve user's search history"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Debug print
            print(f"Retrieving history for user {user_id}")
            
            # First, check if the user exists
            c.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
            if not c.fetchone():
                print(f"User {user_id} not found")
                return []
            
            # Check if there's any history
            c.execute("SELECT COUNT(*) FROM search_history WHERE user_id = ?", (user_id,))
            count = c.fetchone()[0]
            print(f"Found {count} history entries for user {user_id}")
            
            # Get the history with proper ordering
            c.execute("""SELECT id, query, response, context, timestamp 
                        FROM search_history 
                        WHERE user_id = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?""", (user_id, limit))
            
            history = c.fetchall()
            conn.close()
            
            # Debug print
            print(f"Retrieved {len(history)} history entries")
            
            # Convert context from JSON string back to dict
            formatted_history = []
            for item in history:
                try:
                    context_data = json.loads(item[3]) if item[3] else []
                    formatted_history.append({
                        "id": item[0],
                        "query": item[1],
                        "response": item[2],
                        "context": context_data,
                        "timestamp": item[4]
                    })
                except json.JSONDecodeError as e:
                    print(f"Error decoding context for history item {item[0]}: {str(e)}")
                    continue
            
            return formatted_history
        except Exception as e:
            print(f"Error retrieving user history: {str(e)}")
            return []

    def clear_user_history(self, user_id):
        """Clear user's search history"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("DELETE FROM search_history WHERE user_id = ?", (user_id,))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error clearing user history: {str(e)}")
            return False

    def get_similar_queries(self, user_id, current_query, limit=5):
        """Get similar queries from user's history"""
        try:
            history = self.get_user_history(user_id, limit=100)
            if not history:
                return []
            
            # Simple similarity check (can be improved with embeddings)
            similar_queries = []
            for item in history:
                if any(word in current_query.lower() for word in item["query"].lower().split()):
                    similar_queries.append(item)
            
            return similar_queries[:limit]
        except Exception as e:
            return []

def initialize_session_state():
    """Initialize session state variables for user management"""
    if 'user_management' not in st.session_state:
        st.session_state.user_management = UserManagement()
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False
    if 'show_registration' not in st.session_state:
        st.session_state.show_registration = False

def show_login_form():
    """Display login form"""
    with st.form("login_form"):
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            success, result = st.session_state.user_management.login_user(username, password)
            if success:
                st.session_state.user = result
                st.session_state.is_logged_in = True
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(result)
        
        if st.form_submit_button("Register New User"):
            st.session_state.show_registration = True
            st.rerun()

def show_registration_form():
    """Display registration form"""
    with st.form("registration_form"):
        st.subheader("Register New User")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if password != confirm_password:
                st.error("Passwords do not match")
            else:
                success, message = st.session_state.user_management.register_user(
                    username, password, email)
                if success:
                    st.success(message)
                    st.session_state.show_registration = False
                    st.rerun()
                else:
                    st.error(message)
        
        if st.form_submit_button("Back to Login"):
            st.session_state.show_registration = False
            st.rerun()

def show_user_profile():
    """Display user profile and history"""
    if st.session_state.user:
        st.sidebar.subheader(f"Welcome, {st.session_state.user['username']}")
        
        # Add clear history button
        if st.sidebar.button("Clear History"):
            if st.session_state.user_management.clear_user_history(st.session_state.user['user_id']):
                st.sidebar.success("History cleared successfully")
                st.rerun()
            else:
                st.sidebar.error("Failed to clear history")
        
        if st.sidebar.button("Logout"):
            st.session_state.user = None
            st.session_state.is_logged_in = False
            st.rerun()
        
        # Show user history with improved formatting
        st.sidebar.markdown("---")
        st.sidebar.subheader("Search History")
        
        # Get user history
        history = st.session_state.user_management.get_user_history(
            st.session_state.user['user_id'])
        
        # Debug print
        print(f"Displaying history for user {st.session_state.user['user_id']}")
        print(f"Found {len(history)} history entries")
        
        if history:
            for item in history:
                with st.sidebar.container():
                    # Create a unique key for each history item
                    history_key = f"history_{item['id']}"
                    
                    # Display the query and timestamp
                    st.markdown(f"**Query:** {item['query']}")
                    st.markdown(f"**Time:** {item['timestamp']}")
                    
                    # Show response in an expander
                    with st.expander("View Response", expanded=False):
                        st.markdown(item['response'])
                    
                    # Show context in another expander
                    with st.expander("View Context", expanded=False):
                        if item['context']:
                            for ctx in item['context']:
                                st.markdown(f"**Similarity:** {ctx.get('similarity', 'N/A'):.4f}")
                                st.markdown(f"**Question:** {ctx.get('question', 'N/A')}")
                                st.markdown(f"**Answer:** {ctx.get('answer', 'N/A')}")
                                st.markdown("---")
                        else:
                            st.info("No context available")
                    
                    st.markdown("---")
        else:
            st.sidebar.info("No search history available")
            
            # Add debug information
            if st.sidebar.checkbox("Show Debug Info"):
                st.sidebar.write("Debug Information:")
                st.sidebar.write(f"User ID: {st.session_state.user['user_id']}")
                st.sidebar.write(f"Database path: {st.session_state.user_management.db_path}")
                st.sidebar.write("Checking database connection...")
                try:
                    conn = sqlite3.connect(st.session_state.user_management.db_path)
                    c = conn.cursor()
                    c.execute("SELECT COUNT(*) FROM search_history")
                    count = c.fetchone()[0]
                    st.sidebar.write(f"Total history entries in database: {count}")
                    conn.close()
                except Exception as e:
                    st.sidebar.error(f"Database error: {str(e)}") 