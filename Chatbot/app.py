import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import AutonomousVoiceAgent

app = Flask(__name__)
CORS(app)

# Inisialisasi Manual AI Agent lokal
agent = AutonomousVoiceAgent()

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    session_id = data.get('session_id', 'default_user')

    if not user_message:
        return jsonify({'error': 'Pesan tidak boleh kosong'}), 400

    response_text = agent.get_response(user_message, session_id=session_id)
    return jsonify({
        'response': response_text,
        'session_id': session_id
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)