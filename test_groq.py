#!/usr/bin/env python3
"""
Script de prueba para la función ai_insight con Groq
"""
import os
import json
import requests
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración para Groq API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "mixtral-8x7b-32768")

def ai_insight(section, ctx):
    """Generate AI insights using Groq API."""
    if not GROQ_API_KEY:
        return "❌ GROQ_API_KEY no configurada en .env"

    try:
        prompt = f"""Eres analista de datos experto. Explica a personas sin conocimiento técnico en español.
Sección: {section}
Datos: {json.dumps(ctx, ensure_ascii=False, default=str)[:2000]}

Responde EXACTAMENTE con este formato (máximo 150 palabras total):
CONCLUSIÓN: (2-3 oraciones directas y claras)
QUÉ SIGNIFICA: (1-2 oraciones en lenguaje cotidiano, sin jerga)
RECOMENDACIÓN: (1 oración accionable y concreta)"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 400,
            "temperature": 0.7
        }

        print(f"🔄 Conectando a Groq API...")
        print(f"🤖 Usando modelo: {GROQ_MODEL}")

        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        data = response.json()
        result = data["choices"][0]["message"]["content"]

        print("✅ Respuesta recibida de Groq")
        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            return "❌ API key inválida. Verifica tu clave de Groq en https://console.groq.com/"
        elif e.response.status_code == 429:
            return "⏳ Cuota excedida. Espera unos minutos o revisa tu plan en Groq."
        else:
            return f"❌ Error HTTP {e.response.status_code}: {e.response.text[:150]}"
    except requests.exceptions.Timeout:
        return "⏳ La solicitud tardó demasiado. Intenta de nuevo."
    except Exception as e:
        return f"❌ Error de IA: {str(e)[:150]}"

if __name__ == "__main__":
    # Prueba de ejemplo
    test_data = {
        "registros": 100,
        "columnas": 5,
        "tipo": "prueba"
    }

    print("🧪 Probando función ai_insight con Groq...")
    result = ai_insight("Prueba de conexión", test_data)
    print(f"\n📝 Resultado:\n{result}")