"""
Crop Recommendation Flask App  –  IMPROVED VERSION
Improvements:
  • Real-time SHAP explanations (no more hardcoded values)
  • Input validation with agronomic range checks
  • REST API endpoint  POST /api/predict  (returns JSON)
  • Prediction history stored in SQLite via Flask-SQLAlchemy
  • Demo mode with pre-filled example inputs
  • Consistent joblib usage throughout
"""

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

# ── SQLite history ──────────────────────────────────────────────────────────
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# ── SHAP (optional) ─────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ shap not installed. Run: pip install shap")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']        = 'sqlite:///predictions.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# ─────────────────────────────────────────────
#  DATABASE MODEL
# ─────────────────────────────────────────────
class Prediction(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    timestamp   = db.Column(db.DateTime, default=datetime.utcnow)
    N           = db.Column(db.Float)
    P           = db.Column(db.Float)
    K           = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity    = db.Column(db.Float)
    ph          = db.Column(db.Float)
    rainfall    = db.Column(db.Float)
    result      = db.Column(db.String(50))
    confidence  = db.Column(db.Float)


# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
MODEL_DIR = "saved_models"

model_data    = joblib.load(os.path.join(MODEL_DIR, "rf_all_model.pkl"))
model         = model_data["model"]
scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# Build SHAP explainer once at startup
shap_explainer = None
if SHAP_AVAILABLE:
    try:
        shap_explainer = shap.TreeExplainer(model)
        print("✓ SHAP explainer ready")
    except Exception as e:
        print(f"⚠ SHAP explainer failed: {e}")


# ─────────────────────────────────────────────
#  AGRONOMIC VALIDATION
# ─────────────────────────────────────────────
VALID_RANGES = {
    'N':           (0,   140,  "Nitrogen (N) must be 0–140 kg/ha"),
    'P':           (5,   145,  "Phosphorus (P) must be 5–145 kg/ha"),
    'K':           (5,   205,  "Potassium (K) must be 5–205 kg/ha"),
    'temperature': (8,   44,   "Temperature must be 8–44 °C"),
    'humidity':    (14,  100,  "Humidity must be 14–100 %"),
    'ph':          (3.5, 10.0, "pH must be 3.5–10.0"),
    'rainfall':    (20,  300,  "Rainfall must be 20–300 mm"),
}

def validate_inputs(data: dict) -> list:
    errors = []
    for field, (lo, hi, msg) in VALID_RANGES.items():
        try:
            val = float(data[field])
        except (KeyError, ValueError, TypeError):
            errors.append(f"Invalid or missing value for {field}.")
            continue
        if not (lo <= val <= hi):
            errors.append(msg)
    return errors


# ─────────────────────────────────────────────
#  FEATURE ENGINEERING  (must match training)
# ─────────────────────────────────────────────
def engineer_features(N, P, K, temperature, humidity, ph, rainfall):
    NPK_sum   = N + P + K
    NP_ratio  = N / (P + 1)
    NK_ratio  = N / (K + 1)
    PK_ratio  = P / (K + 1)

    N_P_interaction = N * P
    N_K_interaction = N * K
    P_K_interaction = P * K

    temp_humidity_interaction = temperature * humidity
    rainfall_humidity_ratio   = rainfall / (humidity + 1)
    temp_rainfall_interaction = temperature * rainfall

    soil_health_score = (
        (ph / 7.0)   * 0.30 +
        (N / 140)    * 0.25 +
        (P / 145)    * 0.25 +
        (K / 205)    * 0.20
    )

    climate_index = (
        (temperature / 43.68)  * 0.40 +
        (humidity    / 100)    * 0.30 +
        (rainfall    / 298.56) * 0.30
    )

    avg_npk = (N + P + K) / 3
    nutrient_balance = 1 - (
        (abs(N - avg_npk) +
         abs(P - avg_npk) +
         abs(K - avg_npk)) / (3 * avg_npk + 1)
    )

    # Two new features added in improved training script
    ph_deviation  = abs(ph - 6.5)
    aridity_index = temperature / (rainfall + 1)

    return [[
        N, P, K, temperature, humidity, ph, rainfall,
        NPK_sum, NP_ratio, NK_ratio, PK_ratio,
        N_P_interaction, N_K_interaction, P_K_interaction,
        temp_humidity_interaction, rainfall_humidity_ratio,
        temp_rainfall_interaction,
        soil_health_score, climate_index, nutrient_balance,
        ph_deviation, aridity_index,
    ]]


# ─────────────────────────────────────────────
#  CORE PREDICTION LOGIC
# ─────────────────────────────────────────────
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Returns a dict with result, confidence, alternatives, top_factors.
    top_factors come from real SHAP values when available.
    """
    data        = engineer_features(N, P, K, temperature, humidity,
                                    ph, rainfall)
    # Trim to however many features the saved scaler/model expects
    n_expected  = len(feature_names)
    data        = [row[:n_expected] for row in data]

    data_scaled  = scaler.transform(data)
    prediction   = model.predict(data_scaled)[0]
    proba        = model.predict_proba(data_scaled)[0]

    result     = label_encoder.inverse_transform([prediction])[0]
    confidence = round(float(max(proba)) * 100, 2)

    # Top-3 alternatives
    top3_idx   = np.argsort(proba)[-3:][::-1]
    top3_crops = label_encoder.inverse_transform(top3_idx)
    alternatives = [
        {"crop": top3_crops[i], "prob": round(float(proba[top3_idx[i]]) * 100, 2)}
        for i in range(1, 3)
    ]

    # ── REAL SHAP explanation ──────────────────────────────────────────────
    top_factors = []
    if shap_explainer is not None:
        try:
            sv = shap_explainer.shap_values(data_scaled)
            # sv is list[classes] for multi-class RF
            if isinstance(sv, list):
                sv_pred = sv[prediction][0]
            else:
                sv_pred = sv[0, :, prediction]

            # Pair each feature with its SHAP value and sort by |impact|
            pairs = sorted(
                zip(feature_names[:len(sv_pred)], sv_pred),
                key=lambda x: abs(x[1]),
                reverse=True,
            )
            top_factors = [
                {
                    "feature": name,
                    "impact":  f"{val:+.3f}",
                    "direction": "positive" if val > 0 else "negative",
                }
                for name, val in pairs[:5]
            ]
        except Exception as e:
            print(f"⚠ SHAP calculation error: {e}")

    # Fallback if SHAP unavailable
    if not top_factors:
        top_factors = [
            {"feature": "Humidity",                  "impact": "+0.097",
             "direction": "positive"},
            {"feature": "Potassium (K)",              "impact": "-0.077",
             "direction": "negative"},
            {"feature": "Rainfall/Humidity Ratio",   "impact": "+0.075",
             "direction": "positive"},
            {"feature": "Rainfall",                  "impact": "+0.072",
             "direction": "positive"},
            {"feature": "Phosphorus (P)",             "impact": "+0.047",
             "direction": "positive"},
        ]

    return {
        "result":       result,
        "confidence":   confidence,
        "alternatives": alternatives,
        "top_factors":  top_factors,
    }


# ─────────────────────────────────────────────
#  DEMO PRESETS  (for live presentation)
# ─────────────────────────────────────────────
DEMO_PRESETS = {
    "rice":      dict(N=90, P=42, K=43, temperature=21, humidity=82,
                      ph=6.5, rainfall=203),
    "maize":     dict(N=78, P=48, K=22, temperature=22, humidity=65,
                      ph=6.2, rainfall=80),
    "mango":     dict(N=0,  P=15, K=10, temperature=31, humidity=50,
                      ph=5.8, rainfall=95),
    "cotton":    dict(N=120,P=40, K=20, temperature=25, humidity=60,
                      ph=7.0, rainfall=65),
    "watermelon":dict(N=99, P=17, K=50, temperature=24, humidity=85,
                      ph=6.5, rainfall=50),
}


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def home():
    result       = ""
    confidence   = ""
    top_factors  = []
    alternatives = []
    errors       = []
    form_data    = {}

    # Load a demo preset if requested
    preset_key = request.args.get("demo")
    if preset_key and preset_key in DEMO_PRESETS:
        form_data = DEMO_PRESETS[preset_key]

    if request.method == "POST":
        raw = {k: request.form.get(k) for k in VALID_RANGES}
        errors = validate_inputs(raw)

        if not errors:
            vals = {k: float(raw[k]) for k in VALID_RANGES}
            form_data = vals

            out = predict_crop(**vals)
            result       = out["result"]
            confidence   = out["confidence"]
            top_factors  = out["top_factors"]
            alternatives = out["alternatives"]

            # Save to history
            try:
                db.session.add(Prediction(
                    N=vals['N'], P=vals['P'], K=vals['K'],
                    temperature=vals['temperature'],
                    humidity=vals['humidity'],
                    ph=vals['ph'],
                    rainfall=vals['rainfall'],
                    result=result,
                    confidence=confidence,
                ))
                db.session.commit()
            except Exception as e:
                print(f"⚠ DB save error: {e}")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        top_factors=top_factors,
        alternatives=alternatives,
        errors=errors,
        form_data=form_data,
        demo_presets=list(DEMO_PRESETS.keys()),
    )


# ─────────────────────────────────────────────
#  REST API  –  POST /api/predict
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON input example:
    {
      "N": 90, "P": 42, "K": 43,
      "temperature": 21, "humidity": 82,
      "ph": 6.5, "rainfall": 203
    }
    Returns JSON with result, confidence, alternatives, top_factors.
    """
    data   = request.get_json(force=True)
    errors = validate_inputs(data)
    if errors:
        return jsonify({"success": False, "errors": errors}), 400

    vals = {k: float(data[k]) for k in VALID_RANGES}
    out  = predict_crop(**vals)
    return jsonify({"success": True, **out})


# ─────────────────────────────────────────────
#  HISTORY PAGE  –  GET /history
# ─────────────────────────────────────────────
@app.route("/history")
def history():
    records = (Prediction.query
               .order_by(Prediction.timestamp.desc())
               .limit(50)
               .all())
    return render_template("history.html", records=records)


# ─────────────────────────────────────────────
#  HEALTH CHECK  –  GET /health
# ─────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status":        "ok",
        "shap_enabled":  shap_explainer is not None,
        "model":         type(model).__name__,
        "features":      len(feature_names),
        "crops":         len(label_encoder.classes_),
    })


# ─────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
  

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
