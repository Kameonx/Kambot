from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, stream_with_context, make_response
import requests
import os
import json
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session

# Create a directory to store user chat histories
CHAT_DIR = 'chat_histories'
if not os.path.exists(CHAT_DIR):
    os.makedirs(CHAT_DIR)

# API key is directly defined here for simplicity
VENICE_API_KEY = "XIXcv0z57ZOeyPtPO7q37s_ktEvLRAz0E8jFaocVbv"

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
    
    response = make_response(render_template('index.html', chat_history=chat_history))
    response.set_cookie('user_id', user_id, max_age=60*60*24*365)  # Set cookie to expire in 1 year
    return response

@app.route('/stream', methods=['POST'])
def stream_response():
    # Get the user ID from session
    user_id = get_user_id()
    
    data = json.loads(request.data)
    message_id = data.get('message_id')
    
    # Load chat history for this specific user
    chat_history = load_chat_history(user_id)

    def generate():
        payload = {
            "venice_parameters": {"include_venice_system_prompt": True},
            "model": "llama-3.3-70b",
            "messages": [
                {"role": "system", "content": "You are Kambot, a cool, calm, and relaxed assistant with an INFP personality. You're chill, laid-back, and easy-going 😎. You speak in a conversational, clear and concise tone with creative metaphors, witty sarcasm, empathetic insights, and you love using emojis."},
                # Include entire conversation history for context
                *[{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
            ],
            "temperature": 1,
            "top_p": 1,
            "n": 1,
            "stream": True,  # Enable streaming
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "parallel_tool_calls": True
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
                timeout=60  # Increase timeout to 60 seconds for longer responses
            ) as response:
                full_response = ""
                print(f"Venice API status: {response.status_code}")
                print(f"Venice API raw response: {response.text if not response.headers.get('content-type', '').startswith('text/event-stream') else '[streaming response]'}")
                # Check if the response was successful
                if response.status_code != 200:
                    error_msg = f"API Error: Status code {response.status_code} | {response.text}"
                    yield f"data: {json.dumps({'content': error_msg, 'full': error_msg, 'error': True})}\n\n"
                    chat_history.append({"role": "assistant", "content": error_msg})
                    save_chat_history(user_id, chat_history)
                    return

                # Always process as streaming if stream=True
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            data = line_text[6:]  # Remove 'data: ' prefix
                            if data == '[DONE]':
                                break
                            try:
                                # Check if data is not empty before parsing
                                if data.strip():
                                    json_data = json.loads(data)
                                    if 'choices' in json_data and len(json_data['choices']) > 0:
                                        delta = json_data['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            content = delta['content'].replace('*', '')
                                            full_response += content
                                            yield f"data: {json.dumps({'content': content, 'full': full_response})}\n\n"
                            except json.JSONDecodeError as e:
                                print(f"JSON decode error: {e}, data: '{data}'")
                                continue
                # Store the complete response in the user's chat history
                if full_response:
                    chat_history.append({"role": "assistant", "content": full_response})
                    save_chat_history(user_id, chat_history)
                    
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            yield f"data: {json.dumps({'content': error_msg, 'full': error_msg, 'error': True})}\n\n"
            # Add error response to history
            chat_history.append({"role": "assistant", "content": error_msg})
            save_chat_history(user_id, chat_history)

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def get_bot_response(chat_history):
    """Get a response from the bot without streaming (for non-AJAX requests)"""
    payload = {
        "venice_parameters": {"include_venice_system_prompt": True},
        "model": "llama-3.3-70b",
        "messages": [
            {"role": "system", "content": "You are Kambot, a cool, calm, and relaxed assistant with an INFP personality. You're chill, laid-back, and easy-going 😎. You speak in a conversational, clear and concise tone with creative metaphors, witty sarcasm, empathetic insights, and you love using emojis."},
            # Include entire conversation history for context
            *[{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
        ],
        "temperature": 0.15,  # Match docs
        "top_p": 0.9,         # Match docs
        "n": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "parallel_tool_calls": True
    }

    headers = {
        "Authorization": f"Bearer {VENICE_API_KEY}",
        "Content-Type": "application/json"
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
            response_text = response_data['choices'][0]['message']['content'].replace('*', '')
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
    
    return redirect(url_for('chat'))

if __name__ == '__main__':
    app.run(debug=True)
