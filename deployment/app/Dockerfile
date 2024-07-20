FROM python:3.10-slim

# Expose port you want your app on
EXPOSE 8080

WORKDIR /app

# Upgrade pip and install requirements
COPY app/requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app files into root
COPY app/app.py app.py
COPY app/app.yaml app.yaml
COPY app/.streamlit .streamlit

# Copy dataset and ML models
COPY datasets datasets
COPY model model

# Run
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]