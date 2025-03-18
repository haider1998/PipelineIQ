FROM python:3.11.8

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8080

# Environment variable to force Streamlit to run on port 8080
ENV PORT=8080

# Command to run the application
CMD streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
