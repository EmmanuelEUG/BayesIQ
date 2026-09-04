# BayesIQ Analytics - Groq Setup

## Groq Setup (Free Cloud AI)

### 1. Get API Key
```bash
# Go to: https://console.groq.com/
# Create a free account
# Copy your API key
```

### 2. Configure the .env file
```bash
# Edit the .env file and paste your API key:
GROQ_API_KEY=your_actual_groq_key
GROQ_MODEL=mixtral-8x7b-32768
```

### 3. Run the application
```bash
cd /home/emmanuel/env
source bin/activate
cd bayesAnalyzer
pip install -r requirements.txt
streamlit run app.py
```

## Available Models on Groq
- `mixtral-8x7b-32768` - Recommended (fast and efficient)
