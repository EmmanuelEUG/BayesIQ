import streamlit as st
import pandas as pd
import numpy as np
import os, json, requests
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              recall_score, precision_score, f1_score)

load_dotenv()

# Configuración para Groq API
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("Por favor, configura la variable de entorno GROQ_API_KEY con tu clave de API de Groq.")
    st.stop()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")  # Modelo por defecto

st.set_page_config(page_title="BayesIQ · Analytics", page_icon="📊",
                   layout="wide", initial_sidebar_state="collapsed")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for k, v in [("active_section", 0), ("df", None), ("ai_insights", {}),
              ("uploaded_csvs", {}), ("active_csv", None), ("active_csv_label", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#080c14!important;color:#e2e8f0!important;}
::-webkit-scrollbar{width:4px;height:4px;}::-webkit-scrollbar-track{background:#0f1520;}::-webkit-scrollbar-thumb{background:#3b82f6;border-radius:2px;}
.stApp{background:#080c14!important;}
.main .block-container{padding:0 2rem 4rem 2rem!important;max-width:1400px!important;}

/* HERO */
.hero{position:relative;padding:2.5rem 0 1.5rem;text-align:center;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-80px;left:50%;transform:translateX(-50%);width:700px;height:340px;background:radial-gradient(ellipse,rgba(59,130,246,0.13) 0%,transparent 70%);pointer-events:none;}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(2rem,5vw,3.4rem);font-weight:800;background:linear-gradient(135deg,#60a5fa 0%,#a78bfa 50%,#34d399 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-0.03em;line-height:1.1;animation:fadeUp .8s ease both;}
.hero-sub{font-size:.82rem;color:#475569!important;margin-top:.45rem;letter-spacing:.13em;text-transform:uppercase;animation:fadeUp .8s .12s ease both;}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:translateY(0);}}

/* STATS */
.stat-row{display:flex;gap:.7rem;flex-wrap:wrap;margin:1rem 0 .2rem;}
.stat-card{flex:1;min-width:100px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.07);border-radius:14px;padding:.9rem .6rem;text-align:center;transition:transform .2s,border-color .2s;}
.stat-card:hover{transform:translateY(-2px);border-color:rgba(59,130,246,.3);}
.stat-val{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;background:linear-gradient(135deg,#60a5fa,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.stat-lbl{font-size:.65rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}

/* COVERFLOW NAV — pure CSS 3D, driven by data-active on wrapper */
.cf-wrap{margin:1rem 0 .5rem;display:flex;flex-direction:column;align-items:center;gap:0;}
.cf-stage{display:flex;align-items:center;justify-content:center;gap:14px;
  perspective:1000px;perspective-origin:50% 55%;padding:14px 0;}
.cf-btn{
  position:relative;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:7px;
  width:152px;height:74px;
  border-radius:16px;cursor:pointer;
  font-family:'Syne',sans-serif;font-weight:700;font-size:11.5px;letter-spacing:.05em;
  border:1.5px solid rgba(255,255,255,.08);
  background:rgba(255,255,255,.03);color:#475569;
  transition:all .42s cubic-bezier(.4,0,.2,1);
  transform-style:preserve-3d;backface-visibility:hidden;
  box-shadow:0 2px 14px rgba(0,0,0,.5);
  outline:none;
}
.cf-btn svg{width:19px;height:19px;stroke-width:1.8;transition:stroke .3s;pointer-events:none;}
.cf-btn span{pointer-events:none;}

/* position classes set by JS */
.cf-btn.pos-active{
  transform:scale(1.18) translateZ(35px)!important;
  background:linear-gradient(135deg,#3b82f6,#6366f1)!important;
  border-color:transparent!important;color:#fff!important;
  box-shadow:0 12px 42px rgba(59,130,246,.52),0 0 0 1px rgba(99,102,241,.28)!important;
  z-index:10;
}
.cf-btn.pos-active svg{stroke:#fff!important;}
.cf-btn.pos-l1{transform:translateX(10px) translateZ(-28px) rotateY(22deg) scale(.87);opacity:.68;z-index:5;}
.cf-btn.pos-l2{transform:translateX(18px) translateZ(-70px) rotateY(36deg) scale(.70);opacity:.36;z-index:2;}
.cf-btn.pos-r1{transform:translateX(-10px) translateZ(-28px) rotateY(-22deg) scale(.87);opacity:.68;z-index:5;}
.cf-btn.pos-r2{transform:translateX(-18px) translateZ(-70px) rotateY(-36deg) scale(.70);opacity:.36;z-index:2;}
.cf-btn.pos-far{transform:translateZ(-100px) scale(.55);opacity:.1;z-index:1;}

/* hover only on non-active */
.cf-btn:not(.pos-active):hover{
  background:rgba(59,130,246,.1)!important;border-color:rgba(59,130,246,.35)!important;
  color:#93c5fd!important;transform:translateY(-4px) scale(1.04)!important;
}
.cf-btn:not(.pos-active):hover svg{stroke:#93c5fd!important;}

@keyframes cfpop{0%{box-shadow:0 0 0 0 rgba(59,130,246,.55);}100%{box-shadow:0 0 0 22px rgba(59,130,246,0);}}
.cf-btn.popping{animation:cfpop .5s ease-out;}

/* SECTION */
.section-panel{background:linear-gradient(160deg,rgba(15,21,35,.95),rgba(10,15,25,.95));border:1px solid rgba(255,255,255,.06);border-radius:20px;padding:2rem;margin-bottom:1.5rem;animation:panelIn .38s cubic-bezier(.4,0,.2,1) both;}
@keyframes panelIn{from{opacity:0;transform:translateX(22px) scale(.988);}to{opacity:1;transform:none;}}
.sec-title{font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;color:#f1f5f9!important;letter-spacing:-.02em;margin-bottom:.18rem;}
.sec-sub{font-size:.78rem;color:#475569;margin-bottom:1.3rem;}

/* BOXES */
.ibox{background:linear-gradient(135deg,rgba(59,130,246,.07),rgba(99,102,241,.05));border:1px solid rgba(59,130,246,.17);border-radius:12px;padding:1.1rem 1.3rem;margin:.7rem 0;position:relative;overflow:hidden;}
.ibox::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#3b82f6,#6366f1,#34d399);}
.ibox strong{color:#93c5fd!important;display:block;margin-bottom:.3rem;font-family:'Syne',sans-serif;}
.ibox p{color:#94a3b8!important;margin:0;line-height:1.7;font-size:.87rem;}
.aibox{background:linear-gradient(135deg,rgba(52,211,153,.06),rgba(16,185,129,.03));border:1px solid rgba(52,211,153,.17);border-radius:14px;padding:1.3rem;margin:.7rem 0;position:relative;}
.aibox::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#10b981,#34d399,#6ee7b7);}
.aibadge{display:inline-flex;align-items:center;gap:5px;background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.2);border-radius:100px;padding:3px 10px;font-size:.67rem;color:#34d399;letter-spacing:.1em;text-transform:uppercase;font-weight:600;margin-bottom:.65rem;font-family:'Syne',sans-serif;}
.aibadge svg{width:10px;height:10px;}
.aibox p{color:#cbd5e1!important;line-height:1.8;font-size:.89rem;margin:0;white-space:pre-wrap;}
.gdiv{height:1px;margin:1.3rem 0;background:linear-gradient(90deg,transparent,rgba(59,130,246,.32),transparent);}
.badge{display:inline-block;padding:2px 9px;border-radius:100px;font-size:.67rem;font-weight:600;font-family:'Syne',sans-serif;letter-spacing:.07em;text-transform:uppercase;}
.bn{background:rgba(59,130,246,.12);color:#60a5fa;border:1px solid rgba(59,130,246,.22);}
.bc{background:rgba(167,139,250,.12);color:#c4b5fd;border:1px solid rgba(167,139,250,.22);}
.bd{background:rgba(52,211,153,.12);color:#6ee7b7;border:1px solid rgba(52,211,153,.22);}

/* METRICS */
[data-testid="stMetric"]{background:rgba(255,255,255,.025)!important;border:1px solid rgba(255,255,255,.07)!important;border-radius:12px!important;padding:.9rem!important;}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif!important;color:#60a5fa!important;}
[data-testid="stMetricLabel"]{color:#64748b!important;font-size:.7rem!important;text-transform:uppercase;letter-spacing:.1em;}

/* BUTTONS — action buttons */
.stButton>button{background:linear-gradient(135deg,#3b82f6,#6366f1)!important;color:white!important;border:none!important;border-radius:100px!important;font-family:'Syne',sans-serif!important;font-weight:600!important;letter-spacing:.05em!important;padding:.5rem 1.5rem!important;transition:all .2s!important;box-shadow:0 4px 14px rgba(59,130,246,.27)!important;}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 6px 22px rgba(59,130,246,.42)!important;}

/* SELECTS */
.stSelectbox>div>div{background:rgba(15,21,35,.8)!important;border-color:rgba(255,255,255,.1)!important;border-radius:10px!important;color:#e2e8f0!important;}
.streamlit-expanderHeader{background:rgba(255,255,255,.025)!important;border-radius:10px!important;color:#94a3b8!important;}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stDecoration"]{display:none!important;}

/* HIDE the streamlit columns/buttons used as nav trigger — fully invisible */
.nav-triggers{position:absolute;left:-9999px;top:-9999px;width:0;height:0;overflow:hidden;pointer-events:none;}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
DATA_DIR = "data"

def scan_csvs():
    if not os.path.exists(DATA_DIR): return []
    return [(f, os.path.join(DATA_DIR, f))
            for f in sorted(os.listdir(DATA_DIR)) if f.lower().endswith(".csv")]

def detect_columns(df):
    info = []
    for col in df.columns:
        nu = df[col].nunique()
        sample = df[col].dropna().head(3).tolist()
        is_date = False
        if df[col].dtype == 'object':
            try:
                if pd.to_datetime(df[col], errors='coerce').notna().sum() / max(len(df),1) > .7:
                    is_date = True
            except: pass
        if is_date:
            cat, badge, tipo = "Fecha", "date", "Cualitativa (Temporal)"
        elif pd.api.types.is_numeric_dtype(df[col]):
            if nu == 2:    cat, badge, tipo = "Binaria Numérica", "bin", "Cuantitativa Discreta"
            elif nu <= 10: cat, badge, tipo = "Numérica Discreta", "num", "Cuantitativa Discreta"
            else:          cat, badge, tipo = "Numérica Continua", "num", "Cuantitativa Continua"
        else:
            if nu == 2:    cat, badge, tipo = "Binaria (Sí/No)", "bin", "Cualitativa Nominal"
            elif nu <= 15: cat, badge, tipo = "Categórica", "cat", "Cualitativa Nominal"
            else:          cat, badge, tipo = "Texto Libre", "cat", "Cualitativa Nominal"
        miss = int(df[col].isna().sum())
        info.append({"Columna": col, "Categoría": cat, "Tipo": tipo, "badge": badge,
                     "Valores únicos": nu, "Nulos": miss,
                     "% Completitud": f"{(1-miss/max(len(df),1))*100:.1f}%",
                     "Muestra": ", ".join(str(s) for s in sample)})
    return info

def get_groups(info):
    num  = [c["Columna"] for c in info if c["badge"] in ("num","bin") and "Numérica" in c["Categoría"]]
    cat  = [c["Columna"] for c in info if c["badge"] in ("cat","bin") and "Numérica" not in c["Categoría"]]
    date = [c["Columna"] for c in info if c["badge"] == "date"]
    return num, cat, date

def dlayout(corr=False):
    """Legend always BELOW chart. corr=True removes bottom legend (heatmap has colorbar)."""
    base = dict(
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='DM Sans', color='#94a3b8', size=11),
        title_font=dict(family='Syne', color='#f1f5f9', size=14),
        margin=dict(l=40, r=20, t=50, b=30),
        colorway=['#3b82f6','#a78bfa','#34d399','#f59e0b','#f87171','#22d3ee']
    )
    if not corr:
        base["legend"] = dict(
            orientation="h", bgcolor='rgba(0,0,0,0)',
            font=dict(size=10, color='#94a3b8'),
            yanchor="top", y=-0.22, xanchor="center", x=0.5
        )
        base["margin"]["b"] = 95
    return base

def ai_insight(section, ctx):
    """Generate AI insights using Groq API."""
    try:
        prompt = f"""Eres un consultor de datos experto explicando a un cliente de negocio que NO sabe nada de estadística, matemáticas o términos técnicos.

IMPORTANTE: 
- Usa lenguaje simple, como si hablaras con un amigo de negocios
- Evita TODOS los términos técnicos: no digas "correlación", "bayes", "probabilidad", "estadística", "algoritmo", etc.
- Explica todo con ejemplos cotidianos y analogías
- Sé muy detallado en cada sección
- El cliente quiere entender QUÉ SIGNIFICA esto para su negocio

Sección: {section}
Datos: {json.dumps(ctx, ensure_ascii=False, default=str)[:2000]}

Responde EXACTAMENTE con este formato (máximo 300 palabras total):

CONCLUSIÓN EJECUTIVA:
(Escribe 2-3 párrafos explicando el resultado principal de manera clara y directa, como si le contaras a un cliente lo que significa para su negocio)

ANÁLISIS DETALLADO:
(Explica paso a paso qué muestran los datos, usando ejemplos del mundo real. Por ejemplo: "Es como cuando ves que en tu tienda, los clientes que compran X también compran Y el 70% de las veces")

INTERPRETACIÓN PARA EL NEGOCIO:
(Explica qué significa esto en términos prácticos. ¿Cómo afecta las decisiones? ¿Qué patrones ves? Usa analogías de la vida diaria)

RECOMENDACIONES CONCRETAS:
(Da 3-4 acciones específicas que el cliente puede tomar mañana mismo. Sé muy práctico: "Mañana puedes...")"""
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 600,
            "temperature": 0.7
        }
        
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
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

STAR = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>'
AIBADGE = f'<div class="aibadge">{STAR} Análisis con IA</div>'

def aibox(key):
    if key in st.session_state.ai_insights:
        txt = st.session_state.ai_insights[key]
        st.markdown(f'<div class="aibox">{AIBADGE}<p>{txt}</p></div>', unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
available_csvs = scan_csvs()
if st.session_state.df is None and available_csvs:
    st.session_state.df = pd.read_csv(available_csvs[0][1])
    st.session_state.active_csv = available_csvs[0][0]
    st.session_state.active_csv_label = f"[Incluido] {available_csvs[0][0]}"
df = st.session_state.df

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">BayesIQ Analytics</div>'
    '<div class="hero-sub">Detección automática · Teorema de Bayes · Modelo Predictivo · IA Ejecutiva</div>'
    '</div>', unsafe_allow_html=True)

# ── DATASET MANAGER ───────────────────────────────────────────────────────────
with st.expander("  Gestionar datasets", expanded=False):
    opts = {}
    for name, path in available_csvs: opts[f"[Incluido] {name}"] = ("file", path)
    for name, frame in st.session_state.uploaded_csvs.items(): opts[f"[Subido] {name}"] = ("df", frame)
    if len(opts) > 1:
        cur = st.session_state.get("active_csv_label", list(opts.keys())[0])
        idx = list(opts.keys()).index(cur) if cur in opts else 0
        sel = st.selectbox("Dataset activo:", list(opts.keys()), index=idx, key="csv_sel")
        if sel != st.session_state.get("active_csv_label"):
            kind, src = opts[sel]
            st.session_state.df = pd.read_csv(src) if kind == "file" else src
            st.session_state.active_csv_label = sel
            st.session_state.active_csv = sel
            st.session_state.ai_insights = {}
            st.rerun()
    st.markdown('<p style="color:#4b5563;font-size:.78rem;margin:.5rem 0 .3rem;text-transform:uppercase;letter-spacing:.08em;font-weight:600">Subir nuevo CSV</p>', unsafe_allow_html=True)
    nf = st.file_uploader("CSV", type="csv", label_visibility="collapsed",
                          key=f"up_{len(st.session_state.uploaded_csvs)}")
    if nf is not None:
        udf = pd.read_csv(nf)
        st.session_state.uploaded_csvs[nf.name] = udf
        st.session_state.df = udf
        st.session_state.active_csv = nf.name
        st.session_state.active_csv_label = f"[Subido] {nf.name}"
        st.session_state.ai_insights = {}
        st.rerun()


# ── MAIN ──────────────────────────────────────────────────────────────────────
if df is not None:
    info = detect_columns(df)
    num_cols, cat_cols, date_cols = get_groups(info)
    targets = cat_cols + num_cols
    albl = st.session_state.get("active_csv", "Dataset")

    # stat cards
    st.markdown(f"""<div class="stat-row">
      <div class="stat-card"><div class="stat-val">{len(df):,}</div><div class="stat-lbl">Registros</div></div>
      <div class="stat-card"><div class="stat-val">{len(df.columns)}</div><div class="stat-lbl">Variables</div></div>
      <div class="stat-card"><div class="stat-val">{len(num_cols)}</div><div class="stat-lbl">Numéricas</div></div>
      <div class="stat-card"><div class="stat-val">{len(cat_cols)}</div><div class="stat-lbl">Categóricas</div></div>
      <div class="stat-card"><div class="stat-val">{len(date_cols)}</div><div class="stat-lbl">Fechas</div></div>
      <div class="stat-card"><div class="stat-val">{df.isna().sum().sum()}</div><div class="stat-lbl">Nulos</div></div>
    </div>
    <p style="color:#374151;font-size:.72rem;text-align:right;margin:0 0 .4rem">
      Analizando: <span style="color:#60a5fa;font-weight:600">{albl}</span></p>""",
                unsafe_allow_html=True)

    # ── COVERFLOW NAV ─────────────────────────────────────────────────────────
    LABELS = ["Dataset", "Exploración", "Tendencias", "Probabilidades", "Predicción"]
    ICONS  = [
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="2" y="5" width="20" height="14" rx="2"/><path d="M2 10h20M8 5v14"/></svg>',
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M3 3v18h18M7 16v-4m4 4V8m4 8v-7"/></svg>',
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M8 12l3-3 3 3 4-4M3 3v18h18"/></svg>',
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M12 3v18M3 9l9-6 9 6M5 14l-2 3h4zm14 0l-2 3h4z"/></svg>',
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8M12 17v4"/><path stroke-linecap="round" d="M7 10c0-2 10-2 10 0"/></svg>',
    ]
    active = st.session_state.active_section

    # Build coverflow HTML — each card is a <button> with id cf0..cf4
    def cf_pos(i):
        d = i - active
        if d == 0:  return "pos-active"
        if d == -1: return "pos-l1"
        if d == -2: return "pos-l2"
        if d == 1:  return "pos-r1"
        if d == 2:  return "pos-r2"
        return "pos-far"

    cf_cards = ""
    for i, (lbl, icon) in enumerate(zip(LABELS, ICONS)):
        cf_cards += f'<button class="cf-btn {cf_pos(i)}" id="cf{i}" onclick="cfGo({i})">{icon}<span>{lbl}</span></button>'

    st.markdown(f"""
<div class="cf-wrap">
  <div class="cf-stage">{cf_cards}</div>
</div>

<script>
// Coverflow animation + triggers hidden Streamlit button
function cfGo(idx) {{
  // visual update immediately
  const btns = document.querySelectorAll('.cf-btn');
  const posMap = ['pos-l2','pos-l1','pos-active','pos-r1','pos-r2'];
  btns.forEach((b, i) => {{
    b.className = 'cf-btn';
    const d = i - idx;
    if (d === 0)       b.classList.add('pos-active', 'popping');
    else if (d === -1) b.classList.add('pos-l1');
    else if (d === -2) b.classList.add('pos-l2');
    else if (d === 1)  b.classList.add('pos-r1');
    else if (d === 2)  b.classList.add('pos-r2');
    else               b.classList.add('pos-far');
  }});
  setTimeout(() => document.getElementById('cf'+idx)?.classList.remove('popping'), 520);
  // trigger the hidden streamlit button
  const stBtn = document.getElementById('stbtn'+idx);
  if (stBtn) stBtn.click();
}}
</script>
""", unsafe_allow_html=True)

    # Hidden Streamlit buttons — absolutely positioned off-screen
    st.markdown('<div class="nav-triggers">', unsafe_allow_html=True)
    nb_cols = st.columns(len(LABELS))
    for i, col in enumerate(nb_cols):
        with col:
            if st.button(LABELS[i], key=f"stbtn{i}", use_container_width=True):
                st.session_state.active_section = i
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)


    # ═══════════════════════════════════════════
    # 0 · DATASET
    # ═══════════════════════════════════════════
    if active == 0:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Estructura del Dataset</div>'
                    '<div class="sec-sub">Tipos de datos detectados automáticamente y calidad de la información</div>',
                    unsafe_allow_html=True)
        bmap = {"num":"Numérica","cat":"Categórica","date":"Fecha / Tiempo","bin":"Sí / No"}
        rows = [{"Columna":c["Columna"],"Tipo de Dato":bmap.get(c["badge"],c["Categoría"]),
                 "Naturaleza":c["Tipo"],"Valores Únicos":c["Valores únicos"],
                 "Nulos":c["Nulos"],"% Completitud":c["% Completitud"],"Muestra":c["Muestra"]}
                for c in info]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     height=min(420, 50+len(rows)*38))
        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
        with st.expander("  Vista previa — primeras 10 filas"):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        if num_cols:
            with st.expander("  Estadísticas descriptivas"):
                st.dataframe(df[num_cols].describe().round(3).reset_index(),
                             use_container_width=True, hide_index=True)
        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
        if st.button("Generar Análisis Ejecutivo del Dataset", key="ai_ds"):
            with st.spinner("Analizando con IA..."):
                st.session_state.ai_insights["dataset"] = ai_insight(
                    "Estructura del Dataset",
                    {"registros":len(df),"columnas":len(df.columns),
                     "tipos":{c["Columna"]:c["Categoría"] for c in info},
                     "nulos":int(df.isna().sum().sum()),
                     "numericas":num_cols,"categoricas":cat_cols})
        aibox("dataset")
        st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # 1 · EXPLORACIÓN
    # ═══════════════════════════════════════════
    elif active == 1:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Exploración de Variables</div>'
                    '<div class="sec-sub">Distribuciones, relaciones y patrones en los datos</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns([1, 2])
        with c1:
            target = st.selectbox("Variable Objetivo", targets, key="e_t")
            feats = [n for n in num_cols if n != target] or [c for c in cat_cols if c != target]
            feat  = st.selectbox("Variable de Análisis", feats, key="e_f") if feats else None
        with c2:
            if feat:
                fig = px.histogram(df, x=feat, color=target, barmode="group",
                                   title=f"Distribución de {feat} por {target}",
                                   labels={feat:feat,"count":"Frecuencia"})
                fig.update_layout(**dlayout())
                fig.update_traces(marker_line_width=0, opacity=0.85)
                st.plotly_chart(fig, use_container_width=True)

        if feat and feat in num_cols:
            st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
            ca, cb = st.columns(2)
            with ca:
                fb = px.box(df, x=target, y=feat, color=target,
                            title=f"Rangos: {feat} por {target}")
                fb.update_layout(**dlayout()); st.plotly_chart(fb, use_container_width=True)
            with cb:
                fv = px.violin(df, x=target, y=feat, color=target, box=True,
                               title=f"Densidad: {feat}")
                fv.update_layout(**dlayout()); st.plotly_chart(fv, use_container_width=True)

        if len(num_cols) >= 2:
            st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
            corr = df[num_cols].corr().round(3)
            fc   = px.imshow(corr, text_auto=True, aspect="auto",
                             color_continuous_scale="RdBu_r",
                             title="Correlación entre Variables Numéricas", zmin=-1, zmax=1)
            fc.update_layout(**dlayout(corr=True))
            fc.update_layout(margin=dict(l=40,r=20,t=50,b=40))
            st.plotly_chart(fc, use_container_width=True)

        if cat_cols:
            st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
            pc = st.selectbox("Variable para gráfica circular", cat_cols, key="pie_c")
            vc = df[pc].value_counts().reset_index(); vc.columns = [pc,"count"]
            fp = px.pie(vc, names=pc, values="count", title=f"Distribución: {pc}", hole=0.42)
            fp.update_layout(**dlayout())
            fp.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
            st.plotly_chart(fp, use_container_width=True)

        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
        if st.button("Generar Análisis Ejecutivo de Exploración", key="ai_exp"):
            with st.spinner("Analizando con IA..."):
                st.session_state.ai_insights["exp"] = ai_insight(
                    "Exploración de Datos",
                    {"objetivo":target,"analisis":feat,
                     "dist_objetivo":df[target].value_counts().to_dict(),
                     "stats":df[feat].describe().to_dict() if feat and feat in num_cols else {}})
        aibox("exp")
        st.markdown('</div>', unsafe_allow_html=True)


    # ═══════════════════════════════════════════
    # 2 · TENDENCIAS
    # ═══════════════════════════════════════════
    elif active == 2:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Tendencias en el Tiempo</div>'
                    '<div class="sec-sub">Cómo evolucionan las variables a lo largo del período registrado</div>',
                    unsafe_allow_html=True)
        days_span = 0
        if date_cols and num_cols:
            dc = date_cols[0]
            dt = df.copy()
            dt[dc] = pd.to_datetime(dt[dc], errors='coerce')
            dt = dt.dropna(subset=[dc]).sort_values(dc).reset_index(drop=True)
            ft = st.selectbox("Variable a observar", num_cols, key="t_f")

            # Aggregate by date to clean overlapping ticks
            dt["_d"] = dt[dc].dt.date
            agg = dt.groupby("_d")[ft].mean().reset_index()
            agg.columns = ["Fecha", ft]
            agg["Tendencia"] = agg[ft].rolling(5, min_periods=1).mean()

            lo = dlayout()
            lo["margin"]["b"] = 100

            fig_t = go.Figure()
            fig_t.add_trace(go.Scatter(x=agg["Fecha"], y=agg[ft], mode='lines+markers',
                name='Promedio diario',
                line=dict(color='#3b82f6',width=2), marker=dict(size=5,color='#3b82f6')))
            fig_t.add_trace(go.Scatter(x=agg["Fecha"], y=agg["Tendencia"], mode='lines',
                name='Tendencia (media móvil)',
                line=dict(color='#f59e0b',width=2.5,dash='dot')))
            fig_t.update_layout(**lo, title=f"Evolución temporal: {ft}",
                                xaxis_title="Fecha", yaxis_title=ft)
            fig_t.update_xaxes(tickangle=-30, nticks=10)
            st.plotly_chart(fig_t, use_container_width=True)

            st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)

            fig_a = go.Figure()
            fig_a.add_trace(go.Scatter(x=agg["Fecha"], y=agg[ft], fill='tozeroy', mode='lines',
                line=dict(color='#6366f1',width=2),
                fillcolor='rgba(99,102,241,0.1)', name=ft))
            fig_a.update_layout(**lo, title=f"Área acumulada: {ft}",
                                xaxis_title="Fecha", yaxis_title=ft)
            fig_a.update_xaxes(tickangle=-30, nticks=10)
            st.plotly_chart(fig_a, use_container_width=True)

            st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
            days_span = (dt[dc].max()-dt[dc].min()).days
            t1,t2,t3,t4 = st.columns(4)
            t1.metric("Primer registro",  dt[dc].min().strftime("%d/%m/%Y"))
            t2.metric("Último registro",  dt[dc].max().strftime("%d/%m/%Y"))
            t3.metric("Período total",    f"{days_span} días")
            t4.metric("Prom. por día",    f"{len(dt)/max(days_span,1):.1f} registros")
        else:
            st.markdown('<div class="ibox"><strong>Sin columna de fecha detectada</strong>'
                        '<p>El dataset no tiene fechas reconocibles (ej: 2024-01-15 o 15/01/2024).</p></div>',
                        unsafe_allow_html=True)

        st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
        if st.button("Generar Análisis Ejecutivo Temporal", key="ai_t"):
            with st.spinner("Analizando con IA..."):
                st.session_state.ai_insights["temp"] = ai_insight(
                    "Tendencias Temporales",
                    {"tiene_fechas":bool(date_cols),"rango_dias":days_span,
                     "columnas_fecha":date_cols})
        aibox("temp")
        st.markdown('</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # 3 · BAYES
    # ═══════════════════════════════════════════
    elif active == 3:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Cálculo de Probabilidades</div>'
                    '<div class="sec-sub">¿Qué tan probable es un evento y cómo cambia esa probabilidad con nueva información?</div>',
                    unsafe_allow_html=True)
        P_A=P_B=P_BA=P_AB=0.0; umbral=0.0; feat_b=None; val_f=None

        if targets and num_cols:
            c1, c2 = st.columns(2)
            with c1:
                tb = st.selectbox("Evento a estudiar",   targets,        key="b_t")
                vf = st.selectbox("Valor del evento", df[tb].unique(),    key="b_v")
            with c2:
                fopts = [n for n in num_cols if n != tb]
                if fopts:
                    fb = st.selectbox("Variable de evidencia", fopts, key="b_f")
                    u  = st.slider("Umbral de evidencia",
                                   float(df[fb].min()), float(df[fb].max()),
                                   float(df[fb].median()), key="b_u")
                    feat_b=fb; val_f=vf; umbral=u
                else:
                    st.warning("No hay variables numéricas adicionales.")

            if feat_b:
                tot = len(df)
                da  = df[df[tb]==val_f]
                db  = df[df[feat_b]>umbral]
                dab = df[(df[tb]==val_f) & (df[feat_b]>umbral)]
                P_A  = len(da)/tot  if tot>0       else 0
                P_B  = len(db)/tot  if tot>0       else 0
                P_BA = len(dab)/len(da) if len(da)>0 else 0
                P_AB = (P_BA*P_A)/P_B  if P_B>0   else 0

                st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("P(A) — Prob. inicial",     f"{P_A:.4f}",  f"{P_A*100:.1f}%")
                m2.metric("P(B) — Evidencia presente",f"{P_B:.4f}",  f"{P_B*100:.1f}%")
                m3.metric("P(B|A) — Si A, ¿ocurre B?",f"{P_BA:.4f}",f"{P_BA*100:.1f}%")
                m4.metric("P(A|B) — Con evidencia",    f"{P_AB:.4f}",f"{P_AB*100:.1f}%")

                st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
                v1, v2 = st.columns(2)
                with v1:
                    fig_bar = go.Figure([
                        go.Bar(name='Sin evidencia (Prior)', x=['Probabilidad'], y=[P_A],
                               marker_color='#6366f1', marker_line_width=0),
                        go.Bar(name='Con evidencia (Posterior)', x=['Probabilidad'], y=[P_AB],
                               marker_color='#3b82f6', marker_line_width=0),
                    ])
                    fig_bar.update_layout(**dlayout(), title="Antes vs Después de la evidencia",
                                          barmode='group', bargap=0.3)
                    st.plotly_chart(fig_bar, use_container_width=True)
                with v2:
                    rng   = np.linspace(float(df[feat_b].min()), float(df[feat_b].max()), 60)
                    posts = []
                    for uu in rng:
                        pb_   = len(df[df[feat_b]>uu])/tot if tot>0 else 0
                        pbga_ = len(df[(df[tb]==val_f)&(df[feat_b]>uu)])/len(da) if len(da)>0 else 0
                        posts.append((pbga_*P_A/pb_) if pb_>0 else 0)
                    fp2 = go.Figure()
                    fp2.add_trace(go.Scatter(x=rng, y=posts, mode='lines', name='P(A|B)',
                        line=dict(color='#34d399',width=2.5),
                        fill='tozeroy', fillcolor='rgba(52,211,153,0.07)'))
                    fp2.add_vline(x=umbral, line_dash="dash", line_color="#f59e0b",
                                  annotation_text=f"Umbral: {umbral:.2f}",
                                  annotation_position="top right")
                    fp2.update_layout(**dlayout(), title="Curva de probabilidad posterior",
                                      xaxis_title=feat_b, yaxis_title="P(A|B)")
                    st.plotly_chart(fp2, use_container_width=True)

                d = "↑ AUMENTA" if P_AB > P_A else "↓ DISMINUYE"
                st.markdown(
                    f'<div class="ibox"><strong>Resultado del Teorema de Bayes</strong>'
                    f'<p>P(A|B) = {P_BA:.4f} × {P_A:.4f} / {P_B:.4f} = '
                    f'<strong style="color:#60a5fa">{P_AB:.4f}</strong><br>'
                    f'La probabilidad de "<em>{val_f}</em>" cambia de <strong>{P_A:.2%}</strong>'
                    f' a <strong>{P_AB:.2%}</strong> cuando {feat_b} supera {umbral:.2f}.'
                    f' &nbsp;{d}</p></div>', unsafe_allow_html=True)
        else:
            st.warning("Se necesitan columnas categóricas y numéricas.")

        if st.button("Generar Análisis Ejecutivo de Probabilidades", key="ai_b"):
            with st.spinner("Analizando con IA..."):
                st.session_state.ai_insights["bayes"] = ai_insight(
                    "Análisis Bayesiano",
                    {"P_A_prior":round(P_A,4),"P_A_posterior":round(P_AB,4),
                     "P_B_verosimilitud":round(P_BA,4),
                     "cambio":"aumento" if P_AB>P_A else "disminución",
                     "evento":str(val_f) if val_f else "",
                     "variable":feat_b or "","umbral":round(umbral,2)})
        aibox("bayes")
        st.markdown('</div>', unsafe_allow_html=True)


    # ═══════════════════════════════════════════
    # 4 · PREDICCIÓN
    # ═══════════════════════════════════════════
    elif active == 4:
        st.markdown('<div class="section-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Modelo de Predicción</div>'
                    '<div class="sec-sub">El modelo aprende de los datos históricos para predecir resultados futuros</div>',
                    unsafe_allow_html=True)
        if num_cols and targets:
            tm = st.selectbox("¿Qué variable quieres predecir?", targets, key="m_t")
            fc = st.multiselect("Variables que usará el modelo", num_cols,
                                default=num_cols[:min(5,len(num_cols))], key="m_f")
            if fc and st.button("Entrenar Modelo de Predicción", key="train"):
                with st.spinner("Entrenando modelo..."):
                    try:
                        X = df[fc].fillna(0); y = df[tm]
                        X_t,X_v,y_t,y_v = train_test_split(X, y, test_size=0.25, random_state=42)
                        mdl = GaussianNB().fit(X_t, y_t); yp = mdl.predict(X_v)
                        plbls = sorted(y_v.unique())
                        cm    = confusion_matrix(y_v, yp, labels=plbls)
                        st.session_state["mr"] = {
                            "acc":  accuracy_score(y_v, yp),
                            "sens": recall_score(y_v, yp, average='weighted', zero_division=0),
                            "prec": precision_score(y_v, yp, average='weighted', zero_division=0),
                            "f1":   f1_score(y_v, yp, average='weighted', zero_division=0),
                            "cm": cm.tolist(), "labels": [str(l) for l in plbls],
                            "tm": tm, "fc": fc, "tr": len(X_t), "te": len(X_v)}
                    except Exception as e:
                        st.error(f"Error al entrenar: {e}")

            if "mr" in st.session_state:
                r = st.session_state["mr"]
                st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Precisión general",   f"{r['acc']:.1%}")
                m2.metric("Sensibilidad",         f"{r['sens']:.1%}")
                m3.metric("Precisión positivos",  f"{r['prec']:.1%}")
                m4.metric("Balance F1",           f"{r['f1']:.1%}")
                st.markdown(
                    f'<div style="display:flex;gap:.6rem;margin:.5rem 0 1rem;flex-wrap:wrap">'
                    f'<span class="badge bn">Entrenamiento: {r["tr"]} filas</span>'
                    f'<span class="badge bc">Prueba: {r["te"]} filas</span>'
                    f'<span class="badge bd">Clases: {len(r["labels"])}</span></div>',
                    unsafe_allow_html=True)

                st.markdown('<div class="gdiv"></div>', unsafe_allow_html=True)
                cm1, cm2 = st.columns(2)
                with cm1:
                    arr = np.array(r["cm"]); n = arr.shape[0]; lbs = r["labels"][:n]
                    fig_cm = px.imshow(arr, text_auto=True, x=lbs, y=lbs,
                        color_continuous_scale=[[0,'#0f1520'],[0.5,'#3b82f6'],[1,'#6366f1']],
                        title="Matriz de Confusión",
                        labels=dict(x="Predicho", y="Real"))
                    fig_cm.update_layout(**dlayout(corr=True))
                    fig_cm.update_layout(margin=dict(l=60,r=20,t=50,b=60))
                    st.plotly_chart(fig_cm, use_container_width=True)
                with cm2:
                    mn = ["Precisión","Sensibilidad","Prec. pos.","F1"]
                    mv = [r["acc"],r["sens"],r["prec"],r["f1"]]
                    fr = go.Figure()
                    fr.add_trace(go.Scatterpolar(r=mv+[mv[0]], theta=mn+[mn[0]],
                        fill='toself', fillcolor='rgba(59,130,246,0.14)',
                        line=dict(color='#3b82f6',width=2.5), name='Métricas'))
                    fr.update_layout(**dlayout(corr=True), title="Resumen de Rendimiento")
                    fr.update_layout(polar=dict(bgcolor='rgba(0,0,0,0)',
                        radialaxis=dict(visible=True, range=[0,1], color='#475569',
                                        tickfont=dict(size=9)),
                        angularaxis=dict(color='#94a3b8')))
                    st.plotly_chart(fr, use_container_width=True)

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=r["acc"]*100,
                    delta={"reference":70,"valueformat":".1f"},
                    title={"text":"Precisión del Modelo (%)","font":{"family":"Syne","color":"#f1f5f9","size":14}},
                    number={"suffix":"%","font":{"family":"Syne","color":"#60a5fa","size":46}},
                    gauge={"axis":{"range":[0,100],"tickcolor":"#475569","tickfont":{"size":10}},
                           "bar":{"color":"#3b82f6"},
                           "bgcolor":"rgba(0,0,0,0)","bordercolor":"rgba(0,0,0,0)",
                           "steps":[{"range":[0,50],"color":"rgba(239,68,68,0.1)"},
                                    {"range":[50,75],"color":"rgba(245,158,11,0.1)"},
                                    {"range":[75,100],"color":"rgba(52,211,153,0.1)"}],
                           "threshold":{"line":{"color":"#34d399","width":3},"thickness":.8,"value":75}}))
                fig_g.update_layout(**dlayout(corr=True), height=280)
                st.plotly_chart(fig_g, use_container_width=True)

                if st.button("Generar Análisis Ejecutivo del Modelo", key="ai_m"):
                    with st.spinner("Analizando con IA..."):
                        st.session_state.ai_insights["modelo"] = ai_insight(
                            "Modelo Predictivo Naive Bayes",
                            {"accuracy":round(r["acc"],4),"sensibilidad":round(r["sens"],4),
                             "precision":round(r["prec"],4),"f1":round(r["f1"],4),
                             "variable_objetivo":r["tm"],"predictoras":r["fc"],
                             "clases":r["labels"]})
                aibox("modelo")
        else:
            st.warning("Se requieren columnas numéricas para entrenar el modelo.")
        st.markdown('</div>', unsafe_allow_html=True)

# ── EMPTY STATE ───────────────────────────────────────────────────────────────
else:
    st.markdown("""
<div style="text-align:center;padding:5rem 2rem;animation:fadeUp 0.8s ease both">
  <div style="width:66px;height:66px;margin:0 auto 1.1rem;background:linear-gradient(135deg,rgba(59,130,246,0.1),rgba(99,102,241,0.1));border-radius:18px;display:flex;align-items:center;justify-content:center;border:1px solid rgba(59,130,246,0.16)">
    <svg xmlns="http://www.w3.org/2000/svg" width="30" height="30" fill="none" viewBox="0 0 24 24" stroke="#60a5fa" stroke-width="1.5">
      <path stroke-linecap="round" stroke-linejoin="round" d="M9 17v-2m3 2v-4m3 4v-6M3 21h18M3 10l9-7 9 7"/>
    </svg>
  </div>
  <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:700;color:#f1f5f9;margin-bottom:.4rem">
    Carga tu dataset para comenzar
  </div>
  <div style="color:#374151;font-size:.86rem;max-width:360px;margin:0 auto;line-height:1.7">
    BayesIQ detecta automáticamente los tipos de variables y genera análisis estadístico con inteligencia artificial.
  </div>
</div>""", unsafe_allow_html=True)