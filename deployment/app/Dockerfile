FROM python:3.10-slim

# Expose port you want your app on
EXPOSE 8080

WORKDIR /app

# Upgrade pip and install requirements
COPY app/requirements.txt requirements.txt
RUN pip install -U pip
RUN pip install -r requirements.txt

# Copy app files into root
COPY /app .

# Copy datasets
# Dataset copies are specific to keep image size small.
COPY datasets/02_staged/processed_paysim.csv datasets/02_staged/processed_paysim.csv

# Copy models
COPY model model

# Run
ENTRYPOINT ["streamlit", "run", "0_🚨_Payments_Fraud_Screener.py", "--server.port=8080", "--server.address=0.0.0.0"]