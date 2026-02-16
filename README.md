# ANPR System Backend

This service handles Automatic Number Plate Recognition and traffic violation detection using AI models.

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- Requirements listed in `requirements.txt`

### Steps
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the API Server:**
   ```bash
   python api.py
   ```
   *Note: On Windows, you can also use `run_api.bat`.*

## Features
- AI-based License Plate Recognition
- Violation Detection (Helmet, Seatbelt, Overloading)
- Blacklist/Security Alert logic
- Interactive API with FastAPI
