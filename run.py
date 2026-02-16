from app import create_app, db
from app.models import User  # make sure this import exists so SQLAlchemy registers the model
from app.extensions  import db, login_manager

app = create_app()

# Create tables at startup
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)