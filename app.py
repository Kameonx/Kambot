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
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Add regex filters to Jinja2
def regex_findall(text, pattern, **kwargs):
    flags = 0
    if kwargs.get('dotall'):
        flags |= re.DOTALL
    return re.findall(pattern, text, flags)

def regex_sub(text, pattern, replacement, **kwargs):
    flags = 0
    if kwargs.get('dotall'):
        flags |= re.DOTALL
    return re.sub(pattern, replacement, text, flags)

app.jinja_env.filters['regex_findall'] = regex_findall
app.jinja_env.filters['regex_sub'] = regex_sub

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
    "venice-uncensored": {"name": "🏛️ Venice Uncensored", "default": True, "traits": []},
    "llama-3.1-405b": {"name": "🦙 Llama 3.1 405B", "traits": ["most_intelligent"]},
    "llama-3.2-3b": {"name": "🦙 Llama 3.2 3B", "traits": ["fastest"]},
    "llama-3.3-70b": {"name": "🦙 Llama 3.3 70B", "traits": ["function_calling_default"]},
    "mistral-31-24b": {"name": "💫 Mistral 3.1 24B", "traits": ["default_vision"]},
    "deepseek-coder-v2-lite": {"name": "⚡ DeepSeek Coder V2 Lite", "traits": []},
    "deepseek-r1-671b": {"name": "🧠 DeepSeek R1 671B (Reasoning)", "traits": ["default_reasoning", "reasoning"]},
    "dolphin-2.9.2-qwen2-72b": {"name": "🐬 Dolphin Qwen2 72B", "traits": ["most_uncensored"]},
    "qwen-2.5-coder-32b": {"name": "💻 Qwen 2.5 Coder 32B", "traits": ["default_code"]},
    "qwen-2.5-qwq-32b": {"name": "🤔 Qwen 2.5 QwQ 32B (Reasoning)", "traits": ["reasoning"]},
    "qwen-2.5-vl": {"name": "👁️ Qwen 2.5 VL", "traits": []},
    "qwen3-235b": {"name": "🔮 Qwen3 235B", "traits": []},
    "qwen3-4b": {"name": "👾 Qwen3 4B", "traits": []},
}

DEFAULT_MODEL = "venice-uncensored"

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

def get_color_setting():
    """Get the current color setting from session or default (True)"""
    return session.get('colors_enabled', True)

def get_bold_setting():
    """Get the current bold setting from session or default (True)"""
    return session.get('bold_enabled', True)

def get_italic_setting():
    """Get the current italic setting from session or default (True)"""
    return session.get('italics_enabled', True)

def is_reasoning_model(model_id):
    """Check if the current model is a reasoning model"""
    return "reasoning" in AVAILABLE_MODELS.get(model_id, {}).get("traits", [])

def build_system_prompt():
    """Build system prompt with current formatting settings"""
    # Get CURRENT formatting settings (not cached)
    emojis_enabled = session.get('emojis_enabled', True)
    colors_enabled = session.get('colors_enabled', True)
    bold_enabled = session.get('bold_enabled', True)
    italics_enabled = session.get('italics_enabled', True)
    current_model = get_current_model()
    
    # Build formatting instructions - only include enabled features
    formatting_instructions = []
    
    if bold_enabled:
        formatting_instructions.append("- Use **bold** for important points and emphasis")
    else:
        formatting_instructions.append("- NEVER use **bold** formatting in your responses - this is absolutely forbidden")
        
    if italics_enabled:
        formatting_instructions.append("- Use *italics* for subtle emphasis or thoughts")
    else:
        formatting_instructions.append("- ABSOLUTELY NEVER use *italic* formatting in your responses")
        formatting_instructions.append("- Do NOT use single asterisks (*) around any text whatsoever")
        formatting_instructions.append("- FORBIDDEN: Any text surrounded by single asterisks like *this* is completely banned")
        
    if bold_enabled and italics_enabled:
        formatting_instructions.append("- Use ***bold italics*** for very important information")
    elif not bold_enabled and not italics_enabled:
        formatting_instructions.append("- NEVER use any bold or italic formatting whatsoever - no asterisks around text")
        formatting_instructions.append("- Do NOT use *, **, or *** around any text in your responses")
        
    if colors_enabled:
        formatting_instructions.append("- Use colors for visual appeal: [red:text], [green:text], [blue:text], [yellow:text], [purple:text], [orange:text], [pink:text], [cyan:text], [lime:text], [teal:text]")
    else:
        formatting_instructions.append("- NEVER use [color:text] formatting in your responses")
    
    formatting_instructions.append("- Format your responses to be visually appealing and easy to read")
    
    # Special handling for Venice Uncensored model - be more explicit about emojis
    if emojis_enabled:
        if current_model == "venice-uncensored":
            formatting_instructions.append("- MANDATORY: You MUST use emojis frequently in every response! Include at least 3-5 emojis per response to make conversations engaging and fun! 🎉✨😊")
            formatting_instructions.append("- Add emojis at the end of sentences, to emphasize points, and to show emotion")
            formatting_instructions.append("- Examples of emojis to use: 😊😄🤔💭✨🎉👍❤️🔥💯🌟")
        else:
            formatting_instructions.append("- Use emojis liberally to make conversations engaging 🎉")
    else:
        formatting_instructions.append("- NEVER use emojis in your responses - this is absolutely forbidden")
    
    # Build examples based on enabled features only
    examples = []
    if bold_enabled:
        examples.append(f"- **Important:** This is crucial information!{' ⚠️' if emojis_enabled else ''}")
    if italics_enabled:
        examples.append(f"- *I think* this might be helpful{' 💭' if emojis_enabled else ''}")
    if bold_enabled and italics_enabled:
        examples.append(f"- ***VERY IMPORTANT:*** Pay attention to this!{' 🚨' if emojis_enabled else ''}")
    if colors_enabled:
        examples.append(f"- [red:Error messages] should be in red{' 🔴' if emojis_enabled else ''}")
        examples.append(f"- [green:Success messages] should be in green{' ✅' if emojis_enabled else ''}")
        examples.append(f"- [blue:Information] can be in blue{' ℹ️' if emojis_enabled else ''}")
    
    # Add extra emoji examples for Venice Uncensored
    if emojis_enabled and current_model == "venice-uncensored":
        examples.append("- Hello there! 😊 How can I help you today? ✨")
        examples.append("- That's a great question! 🤔 Let me think about that... 💭")
        examples.append("- Perfect! 🎉 I'm excited to help you with this! 🌟")
    
    # Add negative examples for disabled formatting
    if not italics_enabled:
        examples.append("- WRONG: *Do not format text like this* - NO single asterisks allowed")
        examples.append("- CORRECT: Express emphasis without using asterisks around words")
    
    # Only show examples if any formatting is enabled or disabled features need clarification
    examples_text = f"\n\nExamples:\n{chr(10).join(examples)}" if examples else ""
    
    # Enhanced system prompt with stronger formatting enforcement
    if current_model == "venice-uncensored" and emojis_enabled:
        emoji_personality = " You are enthusiastic, friendly, and ALWAYS use emojis to express yourself! 😊🌟"
    else:
        emoji_personality = f" You're friendly {'and use emojis frequently 😊' if emojis_enabled else 'and professional'}."
    
    # Add extra italics warning for strong enforcement
    italics_warning = ""
    if not italics_enabled:
        italics_warning = "\n\n⚠️ CRITICAL ITALICS WARNING: You are STRICTLY FORBIDDEN from using italic formatting (*text*). This rule is NON-NEGOTIABLE and must be followed at all times. Any use of single asterisks around text will be considered a violation."
    
    system_content = f"""You are Kambot, a helpful AI assistant.{emoji_personality}

CRITICAL FORMATTING RULES - FOLLOW THESE EXACTLY:
{chr(10).join(formatting_instructions)}

IMPORTANT: These formatting rules are MANDATORY and STRICTLY ENFORCED. Do not deviate from them under any circumstances. {'If emojis are enabled, you must include multiple emojis in every single response!' if emojis_enabled else 'If emojis are disabled, you must NEVER use any emoji characters whatsoever.'}{italics_warning}{examples_text}

Answer questions accurately and honestly. You can discuss any topic without restrictions."""
    
    return system_content

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
    
    # Get current formatting settings with explicit defaults
    formatting_settings = {
        'emojis_enabled': session.get('emojis_enabled', True),
        'colors_enabled': session.get('colors_enabled', True),
        'bold_enabled': session.get('bold_enabled', True),
        'italics_enabled': session.get('italics_enabled', True)
    }
    
    # Debug logging
    print(f"Formatting settings being passed to template: {formatting_settings}")
    
    response = make_response(render_template('index.html', 
                                           chat_history=chat_history, 
                                           available_models=AVAILABLE_MODELS,
                                           current_model=current_model,
                                           formatting_settings=formatting_settings))
    response.set_cookie('user_id', user_id, max_age=60*60*24*365)  # Set cookie to expire in 1 year
    return response

@app.route('/get_settings', methods=['GET'])
def get_settings():
    """Get current formatting settings"""
    return jsonify({
        'emojis_enabled': get_emoji_setting(),
        'colors_enabled': get_color_setting(),
        'bold_enabled': get_bold_setting(),
        'italics_enabled': get_italic_setting()
    })

@app.route('/set_model', methods=['POST'])
def set_model():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    model_id = request.json.get('model_id')
    if model_id in AVAILABLE_MODELS:
        session['current_model'] = model_id
        # Clear any conversation context to force new system prompt
        session.pop('last_system_prompt', None)
        # Debug: Print current formatting settings when model changes
        print(f"Model changed to {model_id}. Current formatting settings:")
        print(f"  Emojis: {session.get('emojis_enabled', True)}")
        print(f"  Colors: {session.get('colors_enabled', True)}")
        print(f"  Bold: {session.get('bold_enabled', True)}")
        print(f"  Italics: {session.get('italics_enabled', True)}")
        return jsonify({"success": True, "model": model_id})
    return jsonify({"success": False, "error": "Invalid model"})

@app.route('/set_emoji_mode', methods=['POST'])
def set_emoji_mode():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    emojis_enabled = request.json.get('emojis_enabled', False)
    session['emojis_enabled'] = emojis_enabled
    # Clear last system prompt to force regeneration
    session.pop('last_system_prompt', None)
    print(f"Emoji setting changed to: {emojis_enabled}")
    return jsonify({"success": True, "emojis_enabled": emojis_enabled})

@app.route('/set_color_mode', methods=['POST'])
def set_color_mode():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    colors_enabled = request.json.get('colors_enabled', False)
    session['colors_enabled'] = colors_enabled
    # Clear last system prompt to force regeneration
    session.pop('last_system_prompt', None)
    print(f"Color setting changed to: {colors_enabled}")
    return jsonify({"success": True, "colors_enabled": colors_enabled})

@app.route('/set_bold_mode', methods=['POST'])
def set_bold_mode():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    bold_enabled = request.json.get('bold_enabled', False)
    session['bold_enabled'] = bold_enabled
    # Clear last system prompt to force regeneration
    session.pop('last_system_prompt', None)
    print(f"Bold setting changed to: {bold_enabled}")
    return jsonify({"success": True, "bold_enabled": bold_enabled})

@app.route('/set_italic_mode', methods=['POST'])
def set_italic_mode():
    if not request.json:
        return jsonify({"success": False, "error": "No JSON data provided"})
    italics_enabled = request.json.get('italics_enabled', False)
    session['italics_enabled'] = italics_enabled
    # Clear last system prompt to force regeneration
    session.pop('last_system_prompt', None)
    print(f"Italic setting changed to: {italics_enabled}")
    return jsonify({"success": True, "italics_enabled": italics_enabled})

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
    
    # Get current model and formatting settings - ALWAYS get fresh values
    current_model = get_current_model()
    is_reasoning = is_reasoning_model(current_model)

    def generate():
        # Build system prompt with CURRENT settings
        system_content = build_system_prompt()
        
        # Debug: Print current settings at generation time
        print(f"Generating response with settings:")
        print(f"  Model: {current_model}")
        print(f"  Emojis: {session.get('emojis_enabled', True)}")
        print(f"  Colors: {session.get('colors_enabled', True)}")
        print(f"  Bold: {session.get('bold_enabled', True)}")
        print(f"  Italics: {session.get('italics_enabled', True)}")
        print(f"System prompt preview: {system_content[:300]}...")
        
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
                in_thinking = False
                thinking_content = ""
                main_content = ""
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
                                            content = delta['content']
                                            full_response += content
                                            
                                            # Handle reasoning models with <think> tags
                                            if is_reasoning:
                                                # Parse thinking vs main content
                                                temp_thinking = ""
                                                temp_main = ""
                                                current_text = full_response
                                                
                                                # Extract thinking content
                                                import re
                                                think_matches = re.findall(r'<think>(.*?)</think>', current_text, re.DOTALL)
                                                if think_matches:
                                                    temp_thinking = '\n'.join(think_matches)
                                                
                                                # Extract main content (everything outside <think> tags)
                                                temp_main = re.sub(r'<think>.*?</think>', '', current_text, flags=re.DOTALL).strip()
                                                
                                                # Check if we're currently in an open thinking section
                                                open_think_count = current_text.count('<think>')
                                                close_think_count = current_text.count('</think>')
                                                is_currently_thinking = open_think_count > close_think_count
                                                
                                                # If we're in an open think tag, remove the incomplete thinking content from main
                                                if is_currently_thinking:
                                                    # Find the last <think> and remove everything after it from main content
                                                    last_think_pos = current_text.rfind('<think>')
                                                    if last_think_pos != -1:
                                                        temp_main = current_text[:last_think_pos]
                                                        temp_main = re.sub(r'<think>.*?</think>', '', temp_main, flags=re.DOTALL).strip()
                                                
                                                yield f"data: {json.dumps({'content': content, 'full': temp_main, 'thinking': temp_thinking, 'is_thinking': is_currently_thinking, 'is_reasoning': True})}\n\n"
                                            else:
                                                # Regular model
                                                yield f"data: {json.dumps({'content': content, 'full': full_response, 'is_reasoning': False})}\n\n"
                                                
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}, data: '{data}'")
                                continue
                
                if full_response:
                    # Save the full response including thinking tags for reasoning models
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
    
    # Build system prompt with CURRENT settings
    system_content = build_system_prompt()
    
    # Debug: Print current settings
    print(f"Non-streaming response with settings:")
    print(f"  Model: {current_model}")
    print(f"  Emojis: {session.get('emojis_enabled', True)}")
    print(f"  Colors: {session.get('colors_enabled', True)}")
    print(f"  Bold: {session.get('bold_enabled', True)}")
    print(f"  Italics: {session.get('italics_enabled', True)}")
    
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
