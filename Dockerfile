FROM python:3.10-slim

WORKDIR /app

# Create virtual environment inside container
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 5050

CMD ["python3", "run.py"]
