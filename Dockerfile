# 1. Base Image: A lightweight, official Linux/Python 3.10 environment
FROM python:3.10-slim

# 2. Prevent Python from buffering outputs or writing unnecessary cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the directory inside the virtual container
WORKDIR /app

# 4. Install underlying Linux C++ compilers required for heavy math libraries like Scikit-Learn
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy the lock files first (Docker caches this step to make future rebuilds instant)
COPY requirements.txt pyproject.toml ./

# 6. Install the Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the repository into the container
COPY . .

# 8. Install your custom bloodml package
RUN pip install -e .

# 9. Default Command: When the container boots, automatically run the unit tests to prove the math works
CMD ["pytest", "tests/"]