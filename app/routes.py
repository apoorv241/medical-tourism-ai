from flask import Blueprint, render_template, request, jsonify
import uuid

main = Blueprint("main", __name__)

language_agent = None
clinical_agent = None

def init_routes(lang_agent, clinic_agent):
    global language_agent
    language_agent = lang_agent
    

@main.get("/health")
def health():
    return {"status": "ok"}

@main.get("/")
def home():
    return render_template("home.html")

@main.post("/api/plan")
def plan():
    try:
        data = request.get_json(force=True, silent=True) or {}

        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' parameter"}), 400

        if language_agent is None:
            return jsonify({"error": "Agents not initialized"}), 500

        base_thread_id = "dev-thread"
        lang_thread_id = f"{base_thread_id}:lang"
        language_result = language_agent.invoke(query, lang_thread_id)
        
        if language_result.get("needs_clarification"):
            return jsonify(language_result)

        constraints = language_result.get("constraints") or {}

       
        
    except Exception as e:
        print("Error in /api/plan endpoint:", e)
        return jsonify({"error": str(e)}), 500
