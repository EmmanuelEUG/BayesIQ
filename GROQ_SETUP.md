# BayesIQ Analytics - Configuración con Groq

## 🚀 Configuración de Groq (IA gratuita en la nube)

### 1. Obtener API Key
```bash
# Ve a: https://console.groq.com/
# Crea una cuenta gratuita
# Copia tu API key
```

### 2. Configurar el archivo .env
```bash
# Edita el archivo .env y pega tu API key:
GROQ_API_KEY=tu_clave_real_de_groq
GROQ_MODEL=mixtral-8x7b-32768
```

### 3. Ejecutar la aplicación
```bash
cd /home/emmanuel/env
source bin/activate
cd bayesAnalyzer
pip install -r requirements.txt
streamlit run app.py
```

## ⚙️ Modelos disponibles en Groq
- `mixtral-8x7b-32768` - Recomendado (rápido y eficiente)
- `llama2-70b-4096` - Más potente pero más lento
- `gemma-7b-it` - Modelo de Google

## 🆘 Solución de problemas
- **"API key inválida"**: Verifica que la clave en `.env` sea correcta
- **"Cuota excedida"**: Espera unos minutos (cuota gratuita se renueva)
- **"Timeout"**: El servidor está ocupado, intenta de nuevo

## 💡 Ventajas de Groq
- ✅ **Gratuito** con cuota generosa
- ✅ **Muy rápido** (acelerado por hardware especial)
- ✅ **Perfecto para despliegue** en la nube
- ✅ **No requiere instalación** local