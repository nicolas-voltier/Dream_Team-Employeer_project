import streamlit as st
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any
import time

# Import your existing classes
from chatbot_framework import AsyncConversation, Client_prompt_class, Bot_answer_class
from process_graph import GraphProcessor

# Page config
st.set_page_config(
    page_title="Bank of England Docs Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for GPT-like interface
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 2rem;
        color: #1a1a1a;
    }
    
    .main-header h1 {
        color: #1976d2;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #555;
    }
    
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1976d2;
        color: #1a1a1a;
    }
    
    .bot-message {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #388e3c;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #1a1a1a;
    }
    
    .message-timestamp {
        font-size: 0.8rem;
        color: #555;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .message-content {
        font-size: 1rem;
        line-height: 1.5;
        color: #1a1a1a;
        white-space: pre-wrap;
    }
    
    .thinking-indicator {
        display: flex;
        align-items: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .thinking-dots {
        display: inline-block;
        margin-left: 10px;
    }
    
    .thinking-dots::after {
        content: '...';
        animation: thinking 1.5s infinite;
    }
    
    @keyframes thinking {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
    }
    
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #1a1a1a;
    }
    
    .sidebar-info h4 {
        color: #1976d2;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-info p, .sidebar-info li {
        color: #333;
    }
    
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem 1rem;
    }
    
    .stButton > button {
        border-radius: 20px;
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitChatInterface:
    def __init__(self):
        self.graph_processor = GraphProcessor()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "conversation" not in st.session_state:
            st.session_state.conversation = AsyncConversation(
                graph_processor=self.graph_processor,
                client_label="Bank of England employee",
                answerer_label="Chatbot"
            )
        if "agent" not in st.session_state:
            st.session_state.agent = st.session_state.conversation.build_agent(
                llm_model="gpt-5-nano", 
                verbose=False
            )
        if "processing" not in st.session_state:
            st.session_state.processing = False
    
    def display_message(self, message: Dict[str, Any], is_user: bool = True):
        """Display a single message in the chat interface"""
        css_class = "user-message" if is_user else "bot-message"
        role = "You" if is_user else "Assistant"
        
        with st.container():
            st.markdown(f"""
            <div class="{css_class}">
                <div class="message-timestamp">
                    <strong>{role}</strong> • {message['timestamp'].strftime('%H:%M:%S')}
                </div>
                <div class="message-content">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    def display_thinking_indicator(self):
        """Display thinking indicator while processing"""
        st.markdown("""
        <div class="thinking-indicator">
            🤖 Assistant is thinking<span class="thinking-dots"></span>
        </div>
        """, unsafe_allow_html=True)
    
    async def process_user_input(self, user_input: str) -> str:
        """Process user input through the conversation system"""
        try:
            # Add user message to conversation
            st.session_state.conversation.all_messages.append(
                Client_prompt_class(text=user_input, timestamp=datetime.now())
            )
            st.session_state.conversation.generate_ready_to_read()
            
            # Create user prompt with conversation history
            user_prompt = f"""
            Historic of conversation:
            {st.session_state.conversation.ready_to_read}
            """
            
            # Process through agent
            from llama_index.core.workflow import Context
            ctx = Context(st.session_state.agent)
            handler = st.session_state.agent.run(
                user_msg=user_prompt, 
                ctx=ctx, 
                max_iterations=10
            )
            
            response = await handler
            
            # Parse response
            response_content = json.loads(response.response.content)["answer"]
            formatted_response = response_content.replace('\\n', '\n')
            
            # Add bot response to conversation
            st.session_state.conversation.all_messages.append(
                Bot_answer_class(text=formatted_response, timestamp=datetime.now())
            )
            
            return formatted_response
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def run_async_process(self, user_input: str) -> str:
        """Wrapper to run async process in Streamlit"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.process_user_input(user_input))
        finally:
            loop.close()
    
    def render_sidebar(self):
        """Render sidebar with information and controls"""
        with st.sidebar:
            st.markdown("### 🏦 Bank of England Docs Assistant")
            
            st.markdown("""
            <div class="sidebar-info">
                <h4>About</h4>
                <p>I'm your Bank of England documents assistant. I can help you:</p>
                <ul>
                    <li>Search through document corpus</li>
                    <li>Find specific information</li>
                    <li>Get document overviews</li>
                    <li>Answer questions based on available documents</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat controls
            st.markdown("### 💬 Chat Controls")
            
            if st.button("🗑️ Clear Chat History", help="Clear all messages"):
                 st.session_state.messages = []
                 st.session_state.conversation = AsyncConversation(
                     graph_processor=self.graph_processor,
                     client_label="Bank of England employee",
                     answerer_label="Chatbot"
                 )
                 # Add welcome message back
                 welcome_message = {
                     'content': "Hi—I'm the team's chatbot. How can I help you today? I can search through Bank of England documents, provide overviews, or answer specific questions based on our document corpus.",
                     'timestamp': datetime.now(),
                     'is_user': False
                 }
                 st.session_state.messages.append(welcome_message)
                 st.rerun()
            
            # Display conversation stats
            if st.session_state.messages:
                st.markdown("### 📊 Chat Statistics")
                user_msgs = len([m for m in st.session_state.messages if m['is_user']])
                bot_msgs = len([m for m in st.session_state.messages if not m['is_user']])
                st.write(f"**Messages:** {len(st.session_state.messages)}")
                st.write(f"**Your messages:** {user_msgs}")
                st.write(f"**Assistant responses:** {bot_msgs}")
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                st.write("**Model Settings**")
                st.info("Currently using: gpt-4o-mini")
                st.write("**Search Settings**")
                st.info("Similarity threshold: 0.6")
    
    def render_main_interface(self):
        """Render the main chat interface"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🏦 Bank of England Docs Assistant</h1>
            <p>Your intelligent assistant for Bank of England documents and information</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Display chat history
            for message in st.session_state.messages:
                self.display_message(message, message['is_user'])
            
            # Show thinking indicator if processing
            if st.session_state.processing:
                self.display_thinking_indicator()
        
        # Input area (fixed at bottom)
        st.markdown("---")
        
        # Create columns for input and button
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message here...",
                key="user_input",
                placeholder="Ask me about Bank of England documents...",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send", key="send_button", use_container_width=True)
        
        # Process input
        if (send_button or user_input) and user_input and not st.session_state.processing:
            # Add user message to display
            st.session_state.messages.append({
                'content': user_input,
                'timestamp': datetime.now(),
                'is_user': True
            })
            
            # Process the response immediately
            with st.spinner("Processing your request..."):
                bot_response = self.run_async_process(user_input)
            
            # Add bot response
            st.session_state.messages.append({
                'content': bot_response,
                'timestamp': datetime.now(),
                'is_user': False
            })
            
            # Rerun to update the interface
            st.rerun()
    
    def run(self):
        """Main run method for the Streamlit app"""
        self.initialize_session_state()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render main interface
        self.render_main_interface()
        
        # Welcome message for new users
        if not st.session_state.messages:
            welcome_message = {
                'content': "Hi—I'm the team's chatbot. How can I help you today? I can search through Bank of England documents, provide overviews, or answer specific questions based on our document corpus.",
                'timestamp': datetime.now(),
                'is_user': False
            }
            st.session_state.messages.append(welcome_message)
            st.rerun()

def main():
    """Main function to run the Streamlit app"""
    chat_interface = StreamlitChatInterface()
    chat_interface.run()

if __name__ == "__main__":
    main()
