import os
import tempfile
import requests  # Import requests to handle image URL downloads
from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

def download_image_from_url(image_url):
    """Downloads the image from a given URL and saves it locally."""
    response = requests.get(image_url)
    if response.status_code == 200:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(response.content)
            return temp_file.name
    else:
        raise Exception("Failed to download image")

def upload_to_gemini(file_path, mime_type=None):
    """Uploads the given file to Gemini."""
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file.uri

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# In-memory storage for chat sessions (use a database for production)
chat_sessions = {}

@app.route('/api/process', methods=['POST'])
def process_image_and_prompt():
    prompt = request.form.get('prompt')
    image_url = request.form.get('image_url')
    session_id = request.form.get('session_id')
    uid = session_id  # Use session_id as uid for this example

    # Handle session reset
    if not session_id or session_id not in chat_sessions:
        session_id = str(len(chat_sessions) + 1)
        chat_sessions[session_id] = model.start_chat(history=[])

    # Check if image_url is provided and download the image
    if image_url:
        try:
            image_path = download_image_from_url(image_url)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    else:
        return jsonify({"error": "Image URL is required."}), 400

    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        temp_file.write(requests.get(image_url).content)
        image_path = temp_file.name

    # Upload the image to Gemini and get the file URI
    file_uri = upload_to_gemini(image_path, mime_type='image/jpeg')

    # Construct the API URL with prompt, file_uri, and uid (URL 1)
    api_url = f"/gemini?prompt={prompt}&url={file_uri}&uid={uid}"

    # Log the constructed URL for debugging
    print(f"Generated API URL: {api_url}")

    # Update the chat session with the image and prompt
    chat_session = chat_sessions[session_id]
    chat_session.history.append({"role": "user", "parts": [file_uri, prompt]})

    # Send the message and get the response from Gemini
    response = chat_session.send_message(prompt)

    # Clean up the temporary file
    os.remove(image_path)

    return jsonify({"response": response.text, "session_id": session_id})

@app.route('/api/query', methods=['GET'])
def query_prompt():
    prompt = request.args.get('prompt')
    session_id = request.args.get('session_id')

    # Check for valid parameters
    if not prompt or not session_id:
        return jsonify({"error": "Valid session ID and prompt are required."}), 400

    uid = session_id  # Use session_id as uid for this example

    # Retrieve the chat session
    chat_session = chat_sessions.get(session_id)

    if not chat_session:
        return jsonify({"error": "Session ID not found."}), 404

    # Construct the API URL with prompt and uid (URL 2)
    api_url = f"/gemini?prompt={prompt}&uid={uid}"

    # Log the constructed URL for debugging
    print(f"Generated API URL: {api_url}")

    # Append the prompt to the chat session's history
    chat_session.history.append({"role": "user", "parts": [prompt]})

    # Send the message and get the response from Gemini
    response = chat_session.send_message(prompt)
    return jsonify({"response": response.text, "session_id": session_id})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
