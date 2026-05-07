"""
ChurnShield AI - OPTION A: Streamlit Cloud (ONNX - No TensorFlow needed)
ANN Deep Learning Model converted to ONNX format
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle, os, warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ChurnShield AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Space Grotesk',sans-serif!important;}
.stApp{background:#060810!important;}
.main .block-container{background:#060810!important;padding:1.5rem 2rem;max-width:1400px;}
body,p,span,div,label,h1,h2,h3{color:#e2e8f0!important;}
#MainMenu,footer,header{visibility:hidden;}
.hero{background:linear-gradient(135deg,#0d1117 0%,#161b22 60%,#0d1117 100%);
  border:1px solid #21262d;border-radius:20px;padding:36px 44px;margin-bottom:28px;}
.hero-title{font-size:2.3rem;font-weight:800;color:#fff!important;margin:0;}
.hero-sub{font-size:0.92rem;color:#8b949e!important;margin:8px 0 18px;}
.badge{display:inline-block;background:rgba(88,166,255,0.1);border:1px solid rgba(88,166,255,0.25);
  color:#58a6ff!important;font-size:0.72rem;font-weight:600;padding:4px 12px;border-radius:99px;margin:0 5px 5px 0;}
.card{background:#0d1117;border:1px solid #21262d;border-radius:14px;padding:22px 24px;margin-bottom:18px;}
.card-title{font-size:0.7rem!important;font-weight:700!important;text-transform:uppercase;
  letter-spacing:1.8px;color:#58a6ff!important;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #21262d;}
.res-churn{background:linear-gradient(135deg,#1a0a0a,#2d1515);border:1px solid #da3633;
  border-radius:14px;padding:26px;text-align:center;box-shadow:0 0 40px rgba(218,54,51,0.15);margin-bottom:16px;}
.res-safe{background:linear-gradient(135deg,#051a10,#0a2d1c);border:1px solid #238636;
  border-radius:14px;padding:26px;text-align:center;box-shadow:0 0 40px rgba(35,134,54,0.15);margin-bottom:16px;}
.res-title{font-size:1.5rem!important;font-weight:800!important;color:#fff!important;margin:8px 0 4px;}
.res-prob{font-size:1rem!important;color:rgba(255,255,255,0.85)!important;}
.res-note{font-size:0.78rem!important;color:rgba(255,255,255,0.55)!important;font-style:italic;margin-top:6px;}
.sbox{background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;
  border-radius:10px;padding:14px;text-align:center;margin-bottom:8px;}
.sbox-val{font-size:1.5rem!important;font-weight:800!important;color:#58a6ff!important;}
.sbox-lbl{font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:0.8px;}
.rrow{display:flex;align-items:center;gap:10px;background:#0d1117;border:1px solid #21262d;
  border-radius:8px;padding:9px 13px;margin-bottom:6px;}
.rdot{width:9px;height:9px;border-radius:50%;flex-shrink:0;}
.dr{background:#da3633;box-shadow:0 0 8px rgba(218,54,51,0.6);}
.dy{background:#d29922;box-shadow:0 0 8px rgba(210,153,34,0.6);}
.dg{background:#238636;box-shadow:0 0 8px rgba(35,134,54,0.6);}
.rlbl{font-size:0.8rem!important;color:#c9d1d9!important;}
.pill-ok{display:inline-block;background:rgba(35,134,54,0.15);border:1px solid #238636;
  color:#3fb950!important;padding:6px 16px;border-radius:99px;font-size:0.78rem;font-weight:700;}
.pill-err{display:inline-block;background:rgba(218,54,51,0.15);border:1px solid #da3633;
  color:#f85149!important;padding:6px 16px;border-radius:99px;font-size:0.78rem;font-weight:700;}
.fb-wrap{margin-bottom:9px;}
.fb-top{display:flex;justify-content:space-between;margin-bottom:4px;}
.fb-name{font-size:0.8rem!important;color:#c9d1d9!important;}
.fb-pct{font-size:0.8rem!important;color:#58a6ff!important;font-weight:700;}
.fb-track{height:5px;background:#161b22;border-radius:3px;}
.fb-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#1f6feb,#58a6ff);}
.mrow{display:flex;gap:10px;margin:12px 0;}
.mt{flex:1;background:#0d1117;border:1px solid #21262d;border-radius:10px;padding:12px;text-align:center;}
.mv{font-size:1.35rem!important;font-weight:800!important;color:#58a6ff!important;}
.ml{font-size:0.63rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:0.8px;}
.kpi-wrap{display:flex;gap:12px;margin-bottom:20px;flex-wrap:wrap;}
.kpi{flex:1;min-width:120px;background:linear-gradient(135deg,#0d1117,#161b22);border:1px solid #30363d;
  border-radius:12px;padding:16px;text-align:center;}
.kpi-val{font-size:1.5rem!important;font-weight:800!important;color:#58a6ff!important;}
.kpi-lbl{font-size:0.63rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:0.8px;margin-top:3px;}
.atable{width:100%;border-collapse:collapse;}
.atable th{background:#161b22;color:#58a6ff!important;font-size:0.7rem;font-weight:700;
  text-transform:uppercase;letter-spacing:0.8px;padding:10px 14px;text-align:left;border-bottom:1px solid #30363d;}
.atable td{padding:10px 14px;color:#c9d1d9!important;font-size:0.83rem;border-bottom:1px solid #161b22;}
.step{display:flex;gap:12px;align-items:flex-start;margin-bottom:10px;}
.snum{width:24px;height:24px;flex-shrink:0;background:linear-gradient(135deg,#1f6feb,#388bfd);
  border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:800;color:white;}
.stit{font-size:0.84rem!important;font-weight:700!important;color:#e6edf3!important;}
.sdsc{font-size:0.74rem!important;color:#8b949e!important;}
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#0d1117;border-radius:10px;padding:4px;border:1px solid #21262d;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:#8b949e!important;font-weight:600!important;
  font-size:0.86rem!important;padding:8px 20px!important;background:transparent!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1f6feb,#388bfd)!important;color:#fff!important;}
.stSelectbox>div>div{background:#0d1117!important;border:1px solid #30363d!important;color:#e6edf3!important;border-radius:8px!important;}
.stSelectbox svg{fill:#8b949e!important;}
.stNumberInput>div>div>input{background:#0d1117!important;border:1px solid #30363d!important;
  color:#e6edf3!important;border-radius:8px!important;font-weight:600!important;}
.stNumberInput button{background:#161b22!important;border-color:#30363d!important;color:#e6edf3!important;}
[data-testid="stWidgetLabel"] p,.stSlider label,.stSelectbox label,.stNumberInput label,.stRadio label{
  color:#8b949e!important;font-size:0.75rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.8px!important;}
.stRadio>div{gap:8px!important;flex-direction:row!important;}
.stRadio>div>label{background:#0d1117!important;border:1px solid #30363d!important;border-radius:8px!important;
  padding:8px 16px!important;color:#e6edf3!important;cursor:pointer!important;font-size:0.83rem!important;}
.stButton>button{background:linear-gradient(135deg,#1f6feb,#388bfd)!important;color:white!important;
  border:none!important;border-radius:10px!important;padding:13px 28px!important;font-size:0.9rem!important;
  font-weight:700!important;width:100%!important;letter-spacing:0.5px!important;
  box-shadow:0 4px 20px rgba(31,111,235,0.35)!important;text-transform:uppercase!important;}
[data-testid="stSidebar"]{background:#060810!important;border-right:1px solid #21262d!important;}
[data-testid="stSidebar"] *{color:#e2e8f0!important;}
.stDownloadButton>button{background:linear-gradient(135deg,#0a6640,#238636)!important;
  color:white!important;border:none!important;border-radius:8px!important;font-weight:700!important;}
[data-testid="stFileUploader"]{background:#0d1117!important;border:2px dashed #30363d!important;border-radius:12px!important;}
.stProgress>div>div{background:linear-gradient(90deg,#1f6feb,#58a6ff)!important;}
</style>""", unsafe_allow_html=True)

plt.rcParams.update({
    'figure.facecolor':'#0d1117','axes.facecolor':'#161b22','axes.edgecolor':'#30363d',
    'axes.labelcolor':'#8b949e','xtick.color':'#8b949e','ytick.color':'#8b949e',
    'text.color':'#c9d1d9','grid.color':'#21262d','grid.alpha':0.5,
    'axes.spines.top':False,'axes.spines.right':False,
})
BLUE='#58a6ff'; RED='#f85149'; GREEN='#3fb950'; PURP='#a371f7'; ORG='#d29922'

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('final_churn_model.keras')
        with open('scaler.pkl','rb') as f: scaler = pickle.load(f)
        with open('feature_columns.pkl','rb') as f: cols = pickle.load(f)
        return model, scaler, cols, True, ""
    except Exception as e:
        return None, None, None, False, str(e)

@st.cache_data(show_spinner=False)
def load_csv():
    for p in ['Churn Modeling.csv','Churn_Modeling.csv','churn_modeling.csv']:
        if os.path.exists(p): return pd.read_csv(p)
    return None

def preprocess(data, scaler, cols):
    bl = np.log1p(data['Balance'])
    row = {
        'CreditScore':data['CreditScore'],
        'Gender':1 if data['Gender']=='Male' else 0,
        'Age':data['Age'],'Tenure':data['Tenure'],'Balance_log':bl,
        'NumOfProducts':data['NumOfProducts'],'HasCrCard':data['HasCrCard'],
        'IsActiveMember':data['IsActiveMember'],'EstimatedSalary':data['EstimatedSalary'],
        'Geography_Germany':1 if data['Geography']=='Germany' else 0,
        'Geography_Spain':1 if data['Geography']=='Spain' else 0,
        'BalancePerProduct':bl/(data['NumOfProducts']+1),
        'AgeBalanceInteract':data['Age']*bl,
        'TenureAgeRatio':data['Tenure']/(data['Age']+1),
    }
    df = pd.DataFrame([row])
    for c in cols:
        if c not in df.columns: df[c]=0
    return scaler.transform(df[cols]).astype(np.float32)

def predict_onnx(sess, X):
    return float(sess.predict(X.astype("float32"), verbose=0)[0][0])

def risk_factors(data):
    f=[]
    if data['Age']>=50: f.append(("Age ≥ 50 — higher risk group",min(1.0,(data['Age']-50)/25)))
    if data['Geography']=='Germany': f.append(("Germany — highest churn geography",0.78))
    if data['IsActiveMember']==0: f.append(("Inactive member status",0.82))
    if data['NumOfProducts']>=3: f.append(("3+ products — over-exposure",0.72))
    elif data['NumOfProducts']==1: f.append(("Only 1 product — low engagement",0.44))
    if data['Balance']==0: f.append(("Zero account balance",0.55))
    if data['CreditScore']<500: f.append(("Low credit score (< 500)",0.63))
    if data['Gender']=='Female': f.append(("Female (↑ churn in dataset)",0.31))
    if not f: f.append(("No major risk signals detected ✓",0.04))
    f.sort(key=lambda x:x[1],reverse=True)
    return f[:5]

def gauge(prob):
    fig,ax=plt.subplots(figsize=(5,2.8),facecolor='#0d1117')
    ax.set_facecolor('#0d1117'); ax.set_xlim(-1.25,1.25); ax.set_ylim(-0.22,1.18); ax.axis('off')
    t=np.linspace(np.pi,0,300)
    ax.plot(np.cos(t),np.sin(t),color='#161b22',linewidth=22,solid_capstyle='round',zorder=1)
    for s,e,c in [(0,.35,GREEN),(.35,.60,ORG),(.60,1,RED)]:
        ts=np.linspace(np.pi-s*np.pi,np.pi-e*np.pi,120)
        ax.plot(np.cos(ts),np.sin(ts),color=c,linewidth=22,solid_capstyle='butt',zorder=2,alpha=0.3)
    col=RED if prob>.6 else(ORG if prob>.35 else GREEN)
    tf2=np.linspace(np.pi,np.pi-prob*np.pi,300)
    ax.plot(np.cos(tf2),np.sin(tf2),color=col,linewidth=22,solid_capstyle='round',zorder=3)
    ax.plot(np.cos(tf2),np.sin(tf2),color=col,linewidth=30,solid_capstyle='round',zorder=2,alpha=0.1)
    ang=np.pi-prob*np.pi
    ax.annotate('',xy=(.78*np.cos(ang),.78*np.sin(ang)),xytext=(0,0),
                arrowprops=dict(arrowstyle='->',color='white',lw=2.5,mutation_scale=16))
    ax.add_patch(plt.Circle((0,0),.09,color='#0d1117',zorder=9))
    ax.add_patch(plt.Circle((0,0),.06,color=col,zorder=10))
    ax.text(0,-.14,f'{prob*100:.1f}%',ha='center',va='top',fontsize=18,fontweight='black',color=col)
    ax.text(-1.12,-.14,'LOW',ha='center',fontsize=7.5,color=GREEN,fontweight='700')
    ax.text(1.12,-.14,'HIGH',ha='center',fontsize=7.5,color=RED,fontweight='700')
    ax.text(0,1.12,'CHURN PROBABILITY',ha='center',fontsize=7.5,color='#8b949e',fontweight='700')
    plt.tight_layout(pad=0.2); return fig

def bar_chart(labels,values,colors,title):
    fig,ax=plt.subplots(figsize=(7,4),facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    bars=ax.bar(labels,values,color=colors,edgecolor='#0d1117',linewidth=2,width=0.5)
    for bar,v in zip(bars,values):
        ax.text(bar.get_x()+bar.get_width()/2,v+max(values)*0.01,f'{v:.1f}%',
                ha='center',fontweight='bold',fontsize=10,color='#e6edf3')
    ax.set_title(title,color='#e6edf3',fontweight='bold',pad=12)
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.tight_layout(); return fig

def hist_chart(df,col,title):
    fig,ax=plt.subplots(figsize=(7,4),facecolor='#0d1117')
    ax.set_facecolor('#161b22')
    for val,color,label in zip([0,1],[GREEN,RED],['Retained','Churned']):
        ax.hist(df[df['Exited']==val][col],bins=30,alpha=0.7,color=color,label=label,edgecolor='#0d1117')
    ax.set_title(title,color='#e6edf3',fontweight='bold',pad=12)
    ax.legend()
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    plt.tight_layout(); return fig

# ── LOAD ─────────────────────────────────────────────────────────────────────
sess, scaler, feat_cols, loaded, load_err = load_model()
df = load_csv()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:18px 0 12px;">
      <div style="font-size:2.8rem;">🛡️</div>
      <div style="font-size:1.1rem;font-weight:800;color:#58a6ff!important;">ChurnShield AI</div>
      <div style="font-size:0.65rem;color:#8b949e!important;text-transform:uppercase;letter-spacing:1px;">ANN Deep Learning</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:10px 0;"></div>
    <p style="font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">📊 Dataset Stats</p>
    <div class="sbox"><div class="sbox-val">10K</div><div class="sbox-lbl">Training Records</div></div>
    <div class="sbox"><div class="sbox-val">~20%</div><div class="sbox-lbl">Dataset Churn Rate</div></div>
    <div class="sbox"><div class="sbox-val">14</div><div class="sbox-lbl">Model Features</div></div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:14px 0;"></div>
    <p style="font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">🚦 Risk Thresholds</p>
    <div class="rrow"><div class="rdot dr"></div><span class="rlbl"><b>High Risk</b> — ≥ 60%</span></div>
    <div class="rrow"><div class="rdot dy"></div><span class="rlbl"><b>Medium Risk</b> — 35–59%</span></div>
    <div class="rrow"><div class="rdot dg"></div><span class="rlbl"><b>Low Risk</b> — &lt; 35%</span></div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:14px 0;"></div>
    <p style="font-size:0.65rem!important;color:#8b949e!important;text-transform:uppercase;letter-spacing:1.5px;font-weight:700;margin-bottom:8px;">⚙️ Model Status</p>
    """, unsafe_allow_html=True)
    if loaded:
        st.markdown('<div class="pill-ok">✅ &nbsp; ANN Model Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="pill-err">❌ &nbsp; Model Not Found</div>', unsafe_allow_html=True)
        st.caption(load_err)
    st.markdown('<div style="height:1px;background:linear-gradient(90deg,transparent,#30363d,transparent);margin:14px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;font-size:0.65rem;color:#8b949e!important;">ANN · TensorFlow · SMOTE<br>Render.com · sklearn · pandas</p>', unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">🛡️ ChurnShield AI</div>
  <div class="hero-sub">Bank Customer Churn Prediction &nbsp;·&nbsp; ANN Deep Learning + SMOTE &nbsp;·&nbsp; Streamlit Cloud</div>
  <span class="badge">🧠 ANN Deep Learning</span>
  <span class="badge">⚖️ SMOTE Balanced</span>
  <span class="badge">🚀 Render.com</span>
  <span class="badge">🛑 EarlyStopping</span>
  <span class="badge">💾 ModelCheckpoint</span>
  <span class="badge">📊 EDA Analytics</span>
</div>""", unsafe_allow_html=True)

tab1,tab2,tab3,tab4 = st.tabs(["🎯  Predict","📊  EDA Dashboard","📁  Batch","ℹ️  About"])

with tab1:
    L,R = st.columns([1.15,0.85], gap="large")
    with L:
        st.markdown('<div class="card"><div class="card-title">👤 Customer Demographics</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        geography = c1.selectbox("🌍 Geography",["France","Germany","Spain"])
        gender    = c2.selectbox("👤 Gender",["Male","Female"])
        age       = c3.slider("🎂 Age",18,92,35)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">💳 Financial Profile</div>', unsafe_allow_html=True)
        c4,c5 = st.columns(2)
        credit_score = c4.slider("📊 Credit Score",350,850,650)
        balance      = c5.number_input("💰 Balance (€)",0.0,250000.0,75000.0,step=1000.0)
        c6,c7 = st.columns(2)
        est_salary   = c6.number_input("💼 Salary (€)",0.0,200000.0,65000.0,step=1000.0)
        num_products = c7.selectbox("📦 # Products",[1,2,3,4],index=1)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">🏦 Account Behaviour</div>', unsafe_allow_html=True)
        c8,c9,c10 = st.columns(3)
        tenure    = c8.slider("📅 Tenure (yrs)",0,10,5)
        has_cc    = c9.radio("💳 Credit Card",[1,0],format_func=lambda x:"✅ Yes" if x else "❌ No")
        is_active = c10.radio("⚡ Active",[1,0],format_func=lambda x:"✅ Yes" if x else "❌ No")
        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🔮  PREDICT CHURN RISK")

    with R:
        st.markdown('<div class="card"><div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)
        if not predict_btn:
            st.markdown("""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px;margin-bottom:16px;">
              <p style="color:#58a6ff!important;font-weight:700;font-size:0.95rem;margin-bottom:10px;">👈 How to use</p>
              <p style="color:#8b949e!important;font-size:0.86rem;line-height:1.9;margin:0;">
              1. Fill in customer details on the left<br>
              2. Click <b style="color:#58a6ff;">PREDICT CHURN RISK</b><br><br>
              You will receive:<br>
              &nbsp;&nbsp;• Churn probability %<br>
              &nbsp;&nbsp;• Risk level (Low / Medium / High)<br>
              &nbsp;&nbsp;• Visual gauge chart<br>
              &nbsp;&nbsp;• Top risk factors
              </p>
            </div>
            <div class="mrow">
              <div class="mt"><div class="mv">ANN</div><div class="ml">Model Type</div></div>
              <div class="mt"><div class="mv">~87%</div><div class="ml">Accuracy</div></div>
              <div class="mt"><div class="mv">~0.91</div><div class="ml">ROC-AUC</div></div>
            </div>""", unsafe_allow_html=True)
        else:
            if not loaded:
                st.error(f"❌ Model not loaded.\n\n`{load_err}`")
            else:
                cust = {'Geography':geography,'Gender':gender,'Age':age,
                        'CreditScore':credit_score,'Balance':balance,'EstimatedSalary':est_salary,
                        'NumOfProducts':num_products,'Tenure':tenure,'HasCrCard':has_cc,'IsActiveMember':is_active}
                with st.spinner("Analysing..."):
                    Xi   = preprocess(cust, scaler, feat_cols)
                    prob = predict_onnx(sess, Xi)
                churn = prob >= 0.5
                rlv   = "HIGH" if prob>=0.6 else ("MEDIUM" if prob>=0.35 else "LOW")
                if churn:
                    st.markdown(f"""<div class="res-churn">
                      <div style="font-size:2.5rem;">⚠️</div>
                      <p class="res-title">AT RISK OF CHURNING</p>
                      <p class="res-prob">Probability: <b>{prob*100:.1f}%</b> &nbsp;|&nbsp; Risk: <b>{rlv}</b></p>
                      <p class="res-note">Immediate retention action recommended</p>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="res-safe">
                      <div style="font-size:2.5rem;">✅</div>
                      <p class="res-title">LIKELY TO STAY</p>
                      <p class="res-prob">Probability: <b>{prob*100:.1f}%</b> &nbsp;|&nbsp; Risk: <b>{rlv}</b></p>
                      <p class="res-note">Customer appears engaged — monitor regularly</p>
                    </div>""", unsafe_allow_html=True)
                st.pyplot(gauge(prob), use_container_width=True)
                st.markdown('<p style="font-size:0.7rem;color:#8b949e;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin:14px 0 8px;">🔍 Key Risk Factors</p>', unsafe_allow_html=True)
                for name,score in risk_factors(cust):
                    pct=int(score*100)
                    st.markdown(f"""<div class="fb-wrap">
                      <div class="fb-top"><span class="fb-name">{name}</span><span class="fb-pct">{pct}%</span></div>
                      <div class="fb-track"><div class="fb-fill" style="width:{pct}%"></div></div>
                    </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if df is None:
        st.markdown("""<div style="background:#161b22;border:1px solid #d29922;border-radius:12px;
          padding:28px;text-align:center;">
          <div style="font-size:2.5rem;">📂</div>
          <p style="color:#d29922!important;font-weight:700;font-size:1rem;margin:10px 0 6px;">Dataset Not Found</p>
          <p style="color:#8b949e!important;font-size:0.85rem;">Upload Churn Modeling.csv below</p>
        </div>""", unsafe_allow_html=True)
        up = st.file_uploader("Upload Churn Modeling.csv",type=["csv"])
        if up: df = pd.read_csv(up)
        else: st.stop()

    total=len(df); churned=int(df['Exited'].sum()); retained=total-churned; churn_pct=churned/total*100
    st.markdown(f"""<div class="kpi-wrap">
      <div class="kpi"><div class="kpi-val">{total:,}</div><div class="kpi-lbl">Total Customers</div></div>
      <div class="kpi"><div class="kpi-val" style="color:{RED}!important;">{churned:,}</div><div class="kpi-lbl">Churned</div></div>
      <div class="kpi"><div class="kpi-val" style="color:{GREEN}!important;">{retained:,}</div><div class="kpi-lbl">Retained</div></div>
      <div class="kpi"><div class="kpi-val" style="color:{ORG}!important;">{churn_pct:.1f}%</div><div class="kpi-lbl">Churn Rate</div></div>
      <div class="kpi"><div class="kpi-val">{df[df['Exited']==1]['Age'].mean():.0f}</div><div class="kpi-lbl">Avg Churner Age</div></div>
    </div>""", unsafe_allow_html=True)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card"><div class="card-title">📊 Target Distribution</div>', unsafe_allow_html=True)
        fig,axes=plt.subplots(1,2,figsize=(8,4),facecolor='#0d1117')
        counts=df['Exited'].value_counts().sort_index()
        axes[0].bar(['Retained','Churned'],counts.values,color=[GREEN,RED],edgecolor='#0d1117',width=0.45)
        for i,v in enumerate(counts.values): axes[0].text(i,v+50,f'{v:,}',ha='center',fontweight='bold',color='#e6edf3')
        axes[0].set_facecolor('#161b22')
        for sp in axes[0].spines.values(): sp.set_edgecolor('#30363d')
        axes[1].pie(counts.values,labels=['Retained','Churned'],colors=[GREEN,RED],autopct='%1.1f%%',
            startangle=90,wedgeprops=dict(edgecolor='#0d1117',linewidth=2),textprops={'color':'white','fontweight':'bold'})
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">🌍 Churn by Geography</div>', unsafe_allow_html=True)
        geo=df.groupby('Geography')['Exited'].mean()*100
        st.pyplot(bar_chart(geo.index,geo.values,[BLUE,RED,GREEN],'Churn Rate by Country (%)'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="card"><div class="card-title">🎂 Age Distribution</div>', unsafe_allow_html=True)
        st.pyplot(hist_chart(df,'Age','Age Distribution by Churn'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="card"><div class="card-title">📦 Churn by # Products</div>', unsafe_allow_html=True)
        prod=df.groupby('NumOfProducts')['Exited'].mean()*100
        st.pyplot(bar_chart([str(x) for x in prod.index],prod.values,[GREEN,BLUE,ORG,RED][:len(prod)],'Churn Rate by Products (%)'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c5,c6 = st.columns(2)
    with c5:
        st.markdown('<div class="card"><div class="card-title">⚡ Active vs Inactive</div>', unsafe_allow_html=True)
        act=df.groupby('IsActiveMember')['Exited'].mean()*100
        st.pyplot(bar_chart(['Inactive','Active'],act.values,[RED,GREEN],'Churn Rate: Active vs Inactive (%)'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c6:
        st.markdown('<div class="card"><div class="card-title">💰 Balance Distribution</div>', unsafe_allow_html=True)
        st.pyplot(hist_chart(df,'Balance','Balance Distribution by Churn'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    c7,c8 = st.columns(2)
    with c7:
        st.markdown('<div class="card"><div class="card-title">👤 Churn by Gender</div>', unsafe_allow_html=True)
        gc=df.groupby('Gender')['Exited'].mean()*100
        st.pyplot(bar_chart(gc.index,gc.values,[BLUE,PURP],'Churn Rate by Gender (%)'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c8:
        st.markdown('<div class="card"><div class="card-title">📊 Credit Score Distribution</div>', unsafe_allow_html=True)
        st.pyplot(hist_chart(df,'CreditScore','Credit Score by Churn'), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🔥 Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    try:
        from sklearn.preprocessing import LabelEncoder
        d=df.copy(); d['Gender']=LabelEncoder().fit_transform(d['Gender'].astype(str))
        d=pd.get_dummies(d,columns=['Geography'],drop_first=True)
        d.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True,errors='ignore')
        for c in d.columns:
            if d[c].dtype==bool: d[c]=d[c].astype(int)
        d=d.apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='all')
        corr=d.corr(numeric_only=True)
        mask=np.triu(np.ones_like(corr,dtype=bool))
        fig,ax=plt.subplots(figsize=(13,8),facecolor='#0d1117'); ax.set_facecolor('#0d1117')
        sns.heatmap(corr,annot=True,fmt='.2f',cmap='RdBu_r',mask=mask,linewidths=0.5,
                    linecolor='#0d1117',ax=ax,vmin=-1,vmax=1,annot_kws={'size':7.5,'color':'#e6edf3'})
        ax.set_title('Feature Correlation Matrix',color='#e6edf3',fontweight='bold',pad=14,fontsize=13)
        plt.xticks(color='#8b949e',fontsize=8,rotation=45,ha='right')
        plt.yticks(color='#8b949e',fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Heatmap error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card"><div class="card-title">📁 Batch Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8b949e!important;font-size:0.86rem;">Upload CSV with customer data (without Exited column).</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    up2 = st.file_uploader("Upload CSV",type=["csv"])
    if up2:
        if not loaded: st.error("❌ Model not loaded.")
        else:
            df_b=pd.read_csv(up2)
            st.dataframe(df_b.head(5), use_container_width=True)
            if st.button("🚀 RUN BATCH PREDICTION"):
                preds,probs=[],[]
                prog=st.progress(0)
                for i,(_,row) in enumerate(df_b.iterrows()):
                    try:
                        c={'Geography':row.get('Geography','France'),'Gender':row.get('Gender','Male'),
                           'Age':float(row.get('Age',35)),'CreditScore':float(row.get('CreditScore',600)),
                           'Balance':float(row.get('Balance',0)),'EstimatedSalary':float(row.get('EstimatedSalary',60000)),
                           'NumOfProducts':int(row.get('NumOfProducts',1)),'Tenure':float(row.get('Tenure',5)),
                           'HasCrCard':int(row.get('HasCrCard',1)),'IsActiveMember':int(row.get('IsActiveMember',1))}
                        p=predict_onnx(sess, preprocess(c,scaler,feat_cols))
                        probs.append(round(p,4)); preds.append('⚠️ Churn' if p>=0.5 else '✅ Stay')
                    except: probs.append(None); preds.append('❌ Error')
                    prog.progress((i+1)/len(df_b))
                df_b['Churn_Probability']=probs; df_b['Prediction']=preds
                cp=sum('Churn' in p for p in preds)/len(preds)*100
                st.markdown(f"""<div class="mrow">
                  <div class="mt"><div class="mv">{len(df_b)}</div><div class="ml">Total</div></div>
                  <div class="mt"><div class="mv" style="color:{RED}!important;">{cp:.1f}%</div><div class="ml">Churn</div></div>
                  <div class="mt"><div class="mv" style="color:{GREEN}!important;">{100-cp:.1f}%</div><div class="ml">Retain</div></div>
                </div>""", unsafe_allow_html=True)
                st.dataframe(df_b, use_container_width=True)
                st.download_button("⬇️ Download Results",df_b.to_csv(index=False).encode(),"predictions.csv","text/csv")

with tab4:
    c1,c2=st.columns(2,gap="large")
    with c1:
        st.markdown("""<div class="card"><div class="card-title">🧠 Model Architecture</div>
          <table class="atable"><tr><th>Component</th><th>Detail</th></tr>
          <tr><td>Type</td><td>ANN (Deep Learning)</td></tr>
          <tr><td>Hidden Layers</td><td>4 layers (128→64→32→16)</td></tr>
          <tr><td>Activation</td><td>ReLU + Sigmoid (output)</td></tr>
          <tr><td>Regularisation</td><td>Dropout + BatchNorm</td></tr>
          <tr><td>Optimiser</td><td>Adam (lr=0.001)</td></tr>
          <tr><td>Loss</td><td>Binary Cross-Entropy</td></tr>
          <tr><td>Callbacks</td><td>EarlyStopping + ModelCheckpoint</td></tr>
          <tr><td>Imbalance Fix</td><td>SMOTE oversampling</td></tr>
          <tr><td>Cloud Format</td><td>TensorFlow / Keras native</td></tr>
          </table></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">📁 Files in This Repo</div>', unsafe_allow_html=True)
        st.code("""OPTION_B_Render/
├── app.py                   ← This app
├── requirements.txt         ← Dependencies
├── final_churn_model.keras     ← ANN model (ONNX)
├── scaler.pkl               ← StandardScaler
├── feature_columns.pkl      ← Feature list
└── Churn Modeling.csv       ← Dataset (EDA)""","")
        st.markdown('</div>', unsafe_allow_html=True)
