from sqlalchemy.exc import IntegrityError
from flask import Blueprint, render_template, request, jsonify,redirect, url_for, flash
import uuid

from flask_login import LoginManager
from app.models import User
from app.extensions import db

from flask_login import login_user, logout_user, login_required 

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

@main.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    try:
        data = request.form
        email = data.get("email", "").strip().lower()
        first_name = data.get("first_name", "").strip()
        last_name = data.get("last_name", "").strip()
        phone_number = data.get("phone_number", "").strip()
        password = data.get("password", "").strip()

        if not all([email, first_name, last_name, phone_number, password]):
            return render_template("register.html", error="All fields are required")

        # Optional: early checks (nice UX)
        if db.session.query(User).filter_by(email=email).first():
            return render_template("register.html", error="Email already registered")

        if db.session.query(User).filter_by(phone_number=phone_number).first():
            return render_template("register.html", error="Phone number already registered")

        new_user = User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            phone_number=phone_number
        )
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        # ✅ no URL params, message survives redirect
        flash("Successfully registered! Please log in.", "success")
        return redirect(url_for("main.home"))  # <-- home page endpoint

    except IntegrityError:
        db.session.rollback()
        flash("Invalid email or password.", "danger")
        return redirect(url_for("main.home"))
        
    except Exception as e:
        db.session.rollback()
        flash(f"Error in registration: {e}", "danger")
        return redirect(url_for("main.home"))

@main.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or request.form.get("email") or "").strip()
    password = data.get("password") or request.form.get("password") or ""


    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({"error": "invalid credentials"}), 401

    login_user(user)
    return redirect(url_for("main.planner"))  # <-- planner page endpoint


@main.get("/logout")
@login_required
def logout():
    logout_user()
    return jsonify({"message": "logged out"}), 200




@main.get("/planner")
@login_required
def planner():
    return render_template("planner.html")

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
