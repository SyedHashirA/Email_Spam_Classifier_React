
import os
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "model.joblib"))
METRICS_PATH = os.path.join(os.path.dirname(__file__), "models", "metrics.json")

# Lazy load model
_model = None
def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Train first (see README).")
        _model = load(MODEL_PATH)
    return _model

def extract_text_from_pdf(file_storage):
    # Read into memory and parse
    data = file_storage.read()
    reader = PdfReader(io.BytesIO(data))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts).strip()

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/api/metrics", methods=["GET"])
def metrics():
    if os.path.exists(METRICS_PATH):
        import json
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"message": "No metrics saved yet."}), 404

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part 'file' uploaded."}), 400

        f = request.files["file"]
        if f.filename == "":
            return jsonify({"error": "Empty filename."}), 400
        if not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only PDF files are supported."}), 400

        text = extract_text_from_pdf(f)
        if not text or len(text) < 10:
            return jsonify({"error": "Could not extract enough text from the PDF."}), 400

        model = get_model()
        # Pipeline handles vectorization. For probabilities:
        label_idx = model.predict([text])[0]
        # LogisticRegression supports predict_proba; others may not.
        prob = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            prob = float(max(proba))

        label = "SPAM" if int(label_idx) == 1 else "Non-SPAM"
        return jsonify({"label": label, "prob": prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5002"))
    app.run(host="0.0.0.0", port=port, debug=True)
