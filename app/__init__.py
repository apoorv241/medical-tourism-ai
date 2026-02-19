import os
from urllib.parse import quote_plus
from flask import Flask
from dotenv import load_dotenv
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from .extensions import db, login_manager


def create_app():
    load_dotenv()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

    db_user = os.getenv("DB_USER")
    db_pass = quote_plus(os.getenv("DB_PASSWORD", ""))
    db_host = os.getenv("DB_HOST", "mysql")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    login_manager.init_app(app)
    
    @login_manager.unauthorized_handler
    def unauthorized():
        flash("Please login.", "warning")
        return redirect(url_for("main.home"))

    from .models import User

    @login_manager.user_loader
    def load_user(user_id: str):
        return db.session.get(User, int(user_id))

    from .routes import main, init_routes
    app.register_blueprint(main)

    from app.agents.hospital_matching_agent import HospitalMatchingAgent
    from app.agents.laguage_graph_detector import LanguageGraphDetector
    from app.core.llm import create_llm

    llm = create_llm()
    language_agent = LanguageGraphDetector(llm=llm)
    hospital_matching_agent =HospitalMatchingAgent(llm=llm)
    

    init_routes(language_agent, hospital_matching_agent)

    return app