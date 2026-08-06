# CineMatch API — Docker image
#
# Builds and serves the FastAPI backend (content-based + collaborative + hybrid
# recommendation engine). Model artifacts are generated during the build so the
# image is self-contained at runtime — no external model registry is required
# for this project's scope (see docs/06_System_Architecture.md §6).
#
# Build: docker build -t cinematch-api .
# Run:   docker run -p 8000:8000 cinematch-api
# (see docs/09_Development_Guide.md §7 for full deployment instructions)

FROM python:3.10-slim

WORKDIR /app

# Install dependencies first so this layer is cached across rebuilds
# unless requirements.txt itself changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project — src/, api/, data/raw/ are all needed to build the
# model artifacts below. app/ (Streamlit) is not used by this image but is
# harmless to include; it is not started by the container.
COPY . .

# Run the ML pipeline once at build time to produce data/processed/ and
# models/*.pkl inside the image, so the API can load them at container
# startup without needing the raw datasets or scikit-learn training to
# happen again at runtime.
RUN python src/data_preprocessing.py \
    && python src/content_based.py \
    && python src/collaborative_filtering.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
