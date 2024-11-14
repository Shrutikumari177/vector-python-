from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

@app.route('/vectorize', methods=['POST'])
def vectorize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate a 384-dimensional vector using SentenceTransformer
    vector = model.encode(text).tolist()

    return jsonify({"vector": vector})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
