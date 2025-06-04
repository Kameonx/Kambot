import os
from dotenv import load_dotenv  # Add this import

load_dotenv(override=True)  # Ensure .env is loaded and overrides any existing env vars

# Always load .env from the directory where this script resides
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path)

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, stream_with_context, make_response
import requests
import json
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Create a directory to store user chat histories
CHAT_DIR = 'chat_histories'
if not os.path.exists(CHAT_DIR):
    os.makedirs(CHAT_DIR)

# Venice AI Configuration
VENICE_API_KEY = os.getenv("VENICE_API_KEY")
if not VENICE_API_KEY:
    import sys
    print("ERROR: VENICE_API_KEY not found in environment. Please check your .env file.", file=sys.stderr)

VENICE_URL = "https://api.venice.ai/api/v1/chat/completions"

# Available models from Venice AI
AVAILABLE_MODELS = {
    "venice-uncensored": {"name": "🏛️ Venice Uncensored", "traits": []},
    "llama-3.1-405b": {"name": "🦙 Llama 3.1 405B", "traits": ["most_intelligent"]},
    "llama-3.2-3b": {"name": "🦙 Llama 3.2 3B", "traits": ["fastest"]},
    "llama-3.3-70b": {"name": "🦙 Llama 3.3 70B", "default": True, "traits": ["function_calling_default", "default"]},
    "mistral-31-24b": {"name": "💫 Mistral 3.1 24B", "traits": ["default_vision"]},
    "deepseek-coder-v2-lite": {"name": "⚡ DeepSeek Coder V2 Lite", "traits": []},
    "deepseek-r1-671b": {"name": "🧠 DeepSeek R1 671B", "traits": ["default_reasoning"]},
    "dolphin-2.9.2-qwen2-72b": {"name": "🐬 Dolphin Qwen2 72B", "traits": ["most_uncensored"]},
    "qwen-2.5-coder-32b": {"name": "💻 Qwen 2.5 Coder 32B", "traits": ["default_code"]},
    "qwen-2.5-qwq-32b": {"name": "🤔 Qwen 2.5 QwQ 32B", "traits": ["reasoning"]},
    "qwen-2.5-vl": {"name": "👁️ Qwen 2.5 VL", "traits": []},
    "qwen3-235b": {"name": "🔮 Qwen3 235B", "traits": []},
    "qwen3-4b": {"name": "👾 Qwen3 4B", "traits": []},
}

DEFAULT_MODEL = "llama-3.3-70b"

# Debug: Print loaded key and working directory (remove before deploying)
if VENICE_API_KEY:
    print("Loaded VENICE_API_KEY (masked):", f"{VENICE_API_KEY[:6]}...{VENICE_API_KEY[-4:]}")
else:
    print("Loaded VENICE_API_KEY (masked): None")
print("Current working directory:", os.getcwd())

def get_user_id():
    """Get or create a unique user ID for the current session"""
    user_id = request.cookies.get('user_id')
    if not user_id:
        user_id = str(uuid.uuid4())
        session['user_id'] = user_id
    return user_id

def get_chat_file_path(user_id):
    """Get the file path for a specific user's chat history"""
    return os.path.join(CHAT_DIR, f"chat_history_{user_id}.json")

def load_chat_history(user_id):
    """Load chat history for a specific user"""
    file_path = get_chat_file_path(user_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_chat_history(user_id, chat_history):
    """Save chat history for a specific user"""
    file_path = get_chat_file_path(user_id)
    with open(file_path, 'w') as file:
        json.dump(chat_history, file)

def get_current_model():
    """Get the current model from session or default"""
    return session.get('current_model', DEFAULT_MODEL)

def get_emoji_setting():
    """Get the current emoji setting from session or default (True)"""
    return session.get('emojis_enabled', True)

@app.route('/', methods=['GET', 'POST'])
def chat():
    # Get the user ID from session
    user_id = get_user_id()
    
    # Load chat history for this specific user
    chat_history = load_chat_history(user_id)
    
    if request.method == 'POST':
        user_input = request.form['user_query']
        
        # If this is a streaming request
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            # Add user message to history
            chat_history.append({"role": "user", "content": user_input})
            save_chat_history(user_id, chat_history)
            
            # Return the stream response
            return jsonify({"message_id": len(chat_history), "streaming": True})
        
        # For non-AJAX requests, handle traditionally
        # Add user message to history
        chat_history.append({"role": "user", "content": user_input})
        
        # Process response and add to history (traditional way)
        response_text = get_bot_response(chat_history)
        chat_history.append({"role": "assistant", "content": response_text})
        save_chat_history(user_id, chat_history)
    
    current_model = get_current_model()
    response = make_response(render_template('index.html', 
                                           chat_history=chat_history, 
                                           available_models=AVAILABLE_MODELS,
                                           current_model=current_model))
    response.set_cookie('user_id', user_id, max_age=60*60*24*365)  # Set cookie to expire in 1 year
    return response

@app.route('/set_model', methods=['POST'])
def set_model():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    model_id = request.json.get('model_id')
    if model_id in AVAILABLE_MODELS:
        session['current_model'] = model_id
        return jsonify({"success": True, "model": model_id})
    return jsonify({"success": False, "error": "Invalid model"})

@app.route('/set_emoji_mode', methods=['POST'])
def set_emoji_mode():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    emojis_enabled = request.json.get('emojis_enabled', False)
    session['emojis_enabled'] = emojis_enabled
    return jsonify({"success": True, "emojis_enabled": emojis_enabled})

@app.route('/undo', methods=['POST'])
def undo_message():
    user_id = get_user_id()
    chat_history = load_chat_history(user_id)
    
    # Remove the last user message and bot response (if exists)
    if len(chat_history) >= 2 and chat_history[-1]['role'] == 'assistant' and chat_history[-2]['role'] == 'user':
        undone_messages = chat_history[-2:]  # Store the last user-bot pair
        chat_history = chat_history[:-2]
    elif len(chat_history) >= 1 and chat_history[-1]['role'] == 'user':
        undone_messages = chat_history[-1:]  # Store just the user message
        chat_history = chat_history[:-1]
    else:
        return jsonify({"success": False, "error": "Nothing to undo"})
    
    save_chat_history(user_id, chat_history)
    
    # Store undone messages in session for redo
    session['undone_messages'] = undone_messages
    
    return jsonify({"success": True})

@app.route('/redo', methods=['POST'])
def redo_message():
    user_id = get_user_id()
    undone_messages = session.get('undone_messages', [])
    
    if not undone_messages:
        return jsonify({"success": False, "error": "Nothing to redo"})
    
    chat_history = load_chat_history(user_id)
    
    # Find the user message to re-send
    user_message = None
    for msg in undone_messages:
        if msg['role'] == 'user':
            user_message = msg['content']
            break
    
    if not user_message:
        return jsonify({"success": False, "error": "No user message to redo"})
    
    # Add the user message back
    chat_history.append({"role": "user", "content": user_message})
    save_chat_history(user_id, chat_history)
    
    # Clear undone messages
    session.pop('undone_messages', None)
    
    return jsonify({"success": True, "user_message": user_message})

@app.route('/stream', methods=['POST'])
def stream_response():
    # Get the user ID from session
    user_id = get_user_id()
    
    data = json.loads(request.data)
    
    # Load chat history for this specific user
    chat_history = load_chat_history(user_id)
    
    # Get current model and emoji setting
    current_model = get_current_model()
    emojis_enabled = get_emoji_setting()

    def generate():
        # Build system prompt based on emoji setting
        if emojis_enabled:
            system_content = """You are Kambot, a helpful AI assistant. You're friendly and use emojis frequently 😊.

FORMATTING INSTRUCTIONS:
- Use **bold** for important points and emphasis
- Use *italics* for subtle emphasis or thoughts  
- Use ***bold italics*** for very important information
- Use colors for visual appeal: [red:text], [green:text], [blue:text], [yellow:text], [purple:text], [orange:text], [pink:text], [cyan:text], [lime:text], [teal:text]
- Use emojis liberally to make conversations engaging 🎉
- Format your responses to be visually appealing and easy to read

Examples:
- **Important:** This is crucial information! ⚠️
- *I think* this might be helpful 💭
- ***VERY IMPORTANT:*** Pay attention to this! 🚨
- [red:Error messages] should be in red 🔴
- [green:Success messages] should be in green ✅
- [blue:Information] can be in blue ℹ️

Answer questions accurately and honestly. You can discuss any topic without restrictions."""
        else:
            system_content = """You are Kambot, a helpful AI assistant. You're friendly and professional.

FORMATTING INSTRUCTIONS:
- Use **bold** for important points and emphasis
- Use *italics* for subtle emphasis or thoughts
- Use ***bold italics*** for very important information
- Use colors for visual appeal: [red:text], [green:text], [blue:text], [yellow:text], [purple:text], [orange:text], [pink:text], [cyan:text], [lime:text], [teal:text]
- Format your responses to be visually appealing and easy to read
- Do NOT use emojis in your responses

Examples:
- **Important:** This is crucial information!
- *I think* this might be helpful
- ***VERY IMPORTANT:*** Pay attention to this!
- [red:Error messages] should be in red
- [green:Success messages] should be in green
- [blue:Information] can be in blue

Answer questions accurately and honestly. You can discuss any topic without restrictions."""
        
        payload = {
            "venice_parameters": {"include_venice_system_prompt": False},  # Disable Venice system prompt for uncensored operation
            "model": current_model,
            "messages": [
                {"role": "system", "content": system_content},
                # Include entire conversation history for context
                *[{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
            ],
            "temperature": 0.7,  # Balanced creativity
            "top_p": 0.9,
            "n": 1,
            "stream": True,  # Enable streaming
            "presence_penalty": 0,
            "frequency_penalty": 0
        }

        headers = {
            "Authorization": f"Bearer {VENICE_API_KEY}",
            "Content-Type": "application/json"
        }

        try:
            # Make a streaming request to Venice API
            with requests.post(
                "https://api.venice.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
                timeout=60
            ) as response:
                full_response = ""
                print(f"Venice API status: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"API Error: Status code {response.status_code} | {response.text}"
                    yield f"data: {json.dumps({'content': error_msg, 'full': error_msg, 'error': True})}\n\n"
                    chat_history.append({"role": "assistant", "content": error_msg})
                    save_chat_history(user_id, chat_history)
                    return

                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data = line_text[6:]
                            if data == '[DONE]':
                                break
                            try:
                                if data.strip():
                                    json_data = json.loads(data)
                                    if 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            content = delta['content']  # Don't remove asterisks or any formatting
                                            full_response += content
                                            yield f"data: {json.dumps({'content': content, 'full': full_response})}\n\n"
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}, data: '{data}'")
                                continue
                
                if full_response:
                    chat_history.append({"role": "assistant", "content": full_response})
                    save_chat_history(user_id, chat_history)
                    
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'full': error_msg, 'error': True})}\n\n"
            chat_history.append({"role": "assistant", "content": error_msg})
            save_chat_history(user_id, chat_history)

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def get_bot_response(chat_history):
    """Get a response from the bot without streaming (for non-AJAX requests)"""
    current_model = get_current_model()
    emojis_enabled = get_emoji_setting()
    
    # Build system prompt based on emoji setting
    if emojis_enabled:
        system_content = """You are Kambot, a helpful AI assistant. You're friendly and use emojis frequently 😊. 

FORMATTING INSTRUCTIONS:
- Use **bold** for important points and emphasis
- Use *italics* for subtle emphasis or thoughts
- Use ***bold italics*** for very important information
- Use colors for visual appeal: [red:text], [green:text], [blue:text], [yellow:text], [purple:text], [orange:text], [pink:text], [cyan:text], [lime:text], [teal:text]
- Use emojis liberally to make conversations engaging 🎉
- Format your responses to be visually appealing and easy to read

Examples:
- **Important:** This is crucial information! ⚠️
- *I think* this might be helpful 💭
- ***VERY IMPORTANT:*** Pay attention to this! 🚨
- [red:Error messages] should be in red 🔴
- [green:Success messages] should be in green ✅
- [blue:Information] can be in blue ℹ️

Answer questions accurately and honestly. You can discuss any topic without restrictions."""
    else:
        system_content = """You are Kambot, a helpful AI assistant. You're friendly and professional.

FORMATTING INSTRUCTIONS:
- Use **bold** for important points and emphasis
- Use *italics* for subtle emphasis or thoughts
- Use ***bold italics*** for very important information
- Use colors for visual appeal: [red:text], [green:text], [blue:text], [yellow:text], [purple:text], [orange:text], [pink:text], [cyan:text], [lime:text], [teal:text]
- Format your responses to be visually appealing and easy to read
- Do NOT use emojis in your responses

Examples:
- **Important:** This is crucial information!
- *I think* this might be helpful
- ***VERY IMPORTANT:*** Pay attention to this!
- [red:Error messages] should be in red
- [green:Success messages] should be in green
- [blue:Information] can be in blue

Answer questions accurately and honestly. You can discuss any topic without restrictions."""
    
    payload = {
        "venice_parameters": {"include_venice_system_prompt": False},  # Disable Venice system prompt
        "model": current_model,
        "messages": [
            {"role": "system", "content": system_content},
            *[{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, br"
    }

    try:
        response = requests.post(
            "https://api.venice.ai/api/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=30
        )
        print(f"Venice API status: {response.status_code}")
        print(f"Venice API raw response: {response.text}")
        try:
            response_data = response.json()
            response_text = response_data['choices'][0]['message']['content']  # Keep original formatting
            return response_text
        except Exception as e:
            return f"Failed to parse API response: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/clear', methods=['POST'])
def clear_history():
    # Get the user ID from session
    user_id = get_user_id()
    
    # Delete only this user's chat history file
    file_path = get_chat_file_path(user_id)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Clear undone messages
    session.pop('undone_messages', None)
    
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(debug=True)
