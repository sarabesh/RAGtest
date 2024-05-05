FROM langchain/langchain

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY kg_extractor.py .
# COPY utils.py .
# COPY chains.py .

EXPOSE 8503

HEALTHCHECK CMD curl --fail http://localhost:8502/_stcore/health

ENTRYPOINT ["streamlit", "run", "kg_extractor.py", "--server.port=8502", "--server.address=0.0.0.0"]
