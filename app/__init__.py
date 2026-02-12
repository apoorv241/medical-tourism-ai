from flask import Flask

from app.agents.clinical_agent import ClinicalScopeSafetyAgent
from app.agents.laguage_graph_detector import LanguageGraphDetector
from app.core.llm import create_llm
from app.routes import main, init_routes


def create_app():
    app = Flask(__name__)

    llm = create_llm()
    language_agent = LanguageGraphDetector(llm=llm)
    clinical_agent = ClinicalScopeSafetyAgent(llm=llm)

    init_routes(language_agent, clinical_agent)
    app.register_blueprint(main)

    return app
