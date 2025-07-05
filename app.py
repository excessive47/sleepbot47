import openai
import sqlite3
import numpy as np
import csv
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

knowledge = []
with open("schlaflabor_faq.csv", newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        knowledge.append((row["Frage"], row["Antwort"]))


def init_db():
    conn = sqlite3.connect("knowledge.db")
    c = conn.cursor()
    c.execute(
        "CREATE TABLE IF NOT EXISTS knowledge (frage TEXT, antwort TEXT, embedding BLOB)"
    )
    c.execute("SELECT COUNT(*) FROM knowledge")
    if c.fetchone()[0] == 0:
        print("Erzeuge Embeddings und befülle Wissensdatenbank...")
        for frage, antwort in knowledge:
            emb = (
                openai.embeddings.create(input=frage, model="text-embedding-3-small")
                .data[0]
                .embedding
            )
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()
            c.execute(
                "INSERT INTO knowledge (frage, antwort, embedding) VALUES (?, ?, ?)",
                (frage, antwort, emb_bytes),
            )
        conn.commit()
        print("Datenbank initialisiert!")
    conn.close()


def get_embedding(text):
    emb = (
        openai.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )
    return np.array(emb, dtype=np.float32)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def find_best_context(question):
    q_emb = get_embedding(question)
    conn = sqlite3.connect("knowledge.db")
    c = conn.cursor()
    c.execute("SELECT frage, antwort, embedding FROM knowledge")
    best_score = 0
    best_context = None
    for frage, antwort, emb_bytes in c.fetchall():
        emb = np.frombuffer(emb_bytes, dtype=np.float32)
        score = cosine_similarity(q_emb, emb)
        if score > best_score:
            best_score = score
            best_context = antwort
    conn.close()
    return best_context if best_score > 0.75 else None  # Threshold ggf. anpassen


def ask_openai(question, context=None):
    prompt = (
        "Du bist ein digitaler Assistent, der ausschließlich Fragen rund um das Schlaflabor beantwortet. "
        "Wenn eine Frage nichts mit dem Schlaflabor zu tun hat, antworte höflich: "
        "'Ich beantworte nur Fragen zum Schlaflabor.'\n"
    )
    if context:
        prompt += f"Nutze folgendes Praxiswissen als Kontext:\n{context}\n\n"
    prompt += f"Frage: {question}\nAntworte kurz und verständlich."
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=200
    )
    return response.choices[0].message.content.strip()


app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Bitte stelle eine Frage."}), 400
    context = find_best_context(question)
    answer = ask_openai(question, context)
    return jsonify({"answer": answer})


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
