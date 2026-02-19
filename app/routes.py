from datetime import datetime
from time import time
from sqlalchemy.exc import IntegrityError
from flask import Blueprint, render_template, request, jsonify,redirect, url_for, flash,session
from flask_login import login_user, logout_user, login_required, current_user
import uuid

from flask_login import LoginManager
from app.models import User
from app.extensions import db


main = Blueprint("main", __name__)


language_agent = None
hospital_agent = None

def init_routes(lang_agent, hosp_agent):
    global language_agent
    global hospital_agent
    language_agent = lang_agent
    hospital_agent = hosp_agent
    

@main.get("/health")
def health():
    return {"status": "ok"}

@main.get("/")
def home():
    if current_user.is_authenticated:
            return redirect(url_for("main.planner"))
    return render_template("home.html")

@main.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        if current_user.is_authenticated:
            return redirect(url_for("main.planner"))
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
        flash("Invalid email or password.", "danger")
        return redirect(url_for("main.home"))

    login_user(user)
    return redirect(url_for("main.planner"))  # <-- planner page endpoint


@main.get("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("main.home"))




@main.get("/planner")
@login_required
def planner():
    session["thread_id"] = f"user:{current_user.get_id()}:{uuid.uuid4().hex}"
    session.pop("plan_state", None)  # optional: reset constraints on refresh
    return render_template("planner.html")

@main.post("/api/plan")
def plan():
    try:
        data = request.get_json(force=True, silent=True) or {}
        query = (data.get("query") or "").strip()
        if not query:
            return jsonify({"error": "Missing 'query' parameter"}), 400

        
        if "thread_id" not in session:
            session["thread_id"] = f"user:{current_user.get_id()}:{uuid.uuid4().hex}"
        base_thread_id = session["thread_id"]
        lang_thread_id = f"{base_thread_id}:lang"
        hosp_thread_id = f"{base_thread_id}:hospital"

        # -----------------------------
        # Session state (your memory)
        # -----------------------------
        state = session.get("plan_state", {})
        constraints = state.get("constraints") or {}
        needs_clarification = bool(state.get("needs_clarification"))
        missing_fields = state.get("missing_fields") or []
        destination_supported = state.get("destination_supported", None)

        # -----------------------------
        # Helpers
        # -----------------------------
        
        
        def adapt_constraints_for_hospital(constraints: dict) -> dict:
            c = dict(constraints or {})

            # Ensure destination is a dict
            dest = c.get("destination") or {}
            if not isinstance(dest, dict):
                dest = {}

            country = (
                dest.get("country")
                or c.get("country")
                or c.get("destination_country")
            )

            # Language agent often outputs preferred_countries
            if not country:
                preferred = c.get("preferred_countries") or []
                if isinstance(preferred, list) and preferred:
                    country = str(preferred[0]).strip() or None

            dest["country"] = country
            dest.setdefault("city", None)

            c["destination"] = dest
            return c
            c = dict(constraints or {})

            dest = c.get("destination") or {}
            if not isinstance(dest, dict):
                dest = {}

            # Map language-agent output -> hospital-agent expected schema
            if not dest.get("country"):
                preferred = c.get("preferred_countries") or []
                if isinstance(preferred, list) and preferred:
                    dest["country"] = str(preferred[0]).strip() or None

            dest.setdefault("city", None)
            c["destination"] = dest
            return c

        def merge_constraints(old: dict, new: dict) -> dict:
            merged = dict(old or {})
            for k, v in (new or {}).items():
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                merged[k] = v
            return merged

        def constraints_complete() -> bool:
            if missing_fields:
                return False

            c2 = adapt_constraints_for_hospital(constraints)
            dest = c2.get("destination") or {}

            has_destination = bool(isinstance(dest, dict) and dest.get("country"))
            has_procedure = bool(c2.get("procedure"))

            return has_destination and has_procedure and (destination_supported is not False)
            # Best signal: language agent already told us nothing is missing
            if missing_fields:
                return False
            # If you want extra safety, enforce a few keys you consider mandatory:
            required_any = ["procedure", "destination", "country", "destination_country"]
            has_destination = any(constraints.get(k) for k in required_any)
            # budget / origin may be optional in your app; enforce if you want:
            # has_budget = bool(constraints.get("max_budget") or constraints.get("budget"))
            # has_origin = bool(constraints.get("origin_city") or constraints.get("from_city"))
            return has_destination and (destination_supported is not False)

        def looks_like_new_request(user_text: str) -> bool:
            t = user_text.lower()
            # If the user is clearly changing constraints / new procedure / new place → re-run language
            triggers = [
                "in ", "to ", "from ", "budget", "usd", "$", "max", "under",
                "lasik", "knee", "hair", "ivf", "dental", "surgery", "procedure",
                "change", "instead", "different", "compare", "another"
            ]
            return any(x in t for x in triggers)

        # -----------------------------
        # Routing logic
        # -----------------------------

        # 1) If we're in a clarification step, ONLY call language agent
        if needs_clarification:
            language_result = language_agent.invoke(query, lang_thread_id)

            constraints = merge_constraints(constraints, language_result.get("constraints") or {})
            state.update({
                "constraints": constraints,
                "needs_clarification": bool(language_result.get("needs_clarification")),
                "missing_fields": language_result.get("missing_fields") or [],
                "clarification_question": language_result.get("clarification_question"),
                
            })
            
            if "destination_supported" in language_result:
                state["destination_supported"] = language_result["destination_supported"]
            
            
            
            session["plan_state"] = state

            if state["needs_clarification"]:
                return jsonify(language_result), 200

            # clarification done → now go hospital
            hosp_constraints = adapt_constraints_for_hospital(constraints)
            hospital_result = hospital_agent.invoke(hosp_constraints, hosp_thread_id)
            return jsonify({
                "needs_clarification": False,
                "clarification_question": None,
                "hospitals": hospital_result.get("hospitals") or hospital_result.get("results") or [],
            }), 200

        # 2) If we already have complete constraints AND user isn't changing them → go hospital directly
        if constraints_complete() and not looks_like_new_request(query):
            hosp_constraints = adapt_constraints_for_hospital(constraints)
            hospital_result = hospital_agent.invoke(hosp_constraints, hosp_thread_id)
            return jsonify({
                "needs_clarification": False,
                "clarification_question": None,
                "hospitals": hospital_result.get("hospitals") or hospital_result.get("results") or [],
            }), 200

        # 3) Otherwise, call language agent (new request OR updating constraints)
        language_result = language_agent.invoke(query, lang_thread_id)

        constraints = merge_constraints(constraints, language_result.get("constraints") or {})
        state.update({
            "constraints": constraints,
            "needs_clarification": bool(language_result.get("needs_clarification")),
            "missing_fields": language_result.get("missing_fields") or [],
            "clarification_question": language_result.get("clarification_question"),
        })

        # Only update destination_supported if explicitly returned
        if "destination_supported" in language_result:
            state["destination_supported"] = language_result["destination_supported"]
        session["plan_state"] = state

        if state["needs_clarification"]:
            return jsonify(language_result), 200

        hosp_constraints = adapt_constraints_for_hospital(constraints)
        hospital_result = hospital_agent.invoke(hosp_constraints, hosp_thread_id)
        return jsonify({
            "needs_clarification": False,
            "clarification_question": None,
            "hospitals": hospital_result.get("hospitals") or hospital_result.get("results") or [],
        }), 200

    except Exception as e:
        print("Error in /api/plan endpoint:", e)
        return jsonify({"error": str(e)}), 500
    