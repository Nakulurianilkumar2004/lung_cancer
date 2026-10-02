"""
Lung Cancer Prediction - Flask App
SHAP feature selection + Neural Encoder latent features + CatBoost

Run:  python app.py   ->  open http://127.0.0.1:5000
"""
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")


def load_pkl(name):
    with open(os.path.join(MODEL_DIR, name), "rb") as f:
        return pickle.load(f)


# ---------------- Load saved components ----------------
model          = load_pkl("catboost_model.pkl")
scaler         = load_pkl("scaler.pkl")
label_encoders = load_pkl("label_encoders.pkl")
encoder_bundle = load_pkl("encoder.pkl")
feature_info   = load_pkl("feature_info.pkl")
metrics        = load_pkl("metrics.pkl")

FEATURES   = feature_info["feature_names"]          # exact column names, in order
TOP_IDX    = np.array(feature_info["top_idx"])
TARGET     = feature_info["target_column"]
YES, NO    = feature_info["symptom_coding"]["YES"], feature_info["symptom_coding"]["NO"]
ENC_SD     = encoder_bundle["state_dict"]
TARGET_LE  = label_encoders.get(TARGET)
INPUT_LES  = {c: le for c, le in label_encoders.items() if c != TARGET}


def encoder_forward(x):
    """NumPy version of: Linear(in,64)-ReLU-Linear(64,32)-ReLU-Linear(32,16)."""
    h = np.maximum(0, x @ ENC_SD["net.0.weight"].T + ENC_SD["net.0.bias"])
    h = np.maximum(0, h @ ENC_SD["net.2.weight"].T + ENC_SD["net.2.bias"])
    return h @ ENC_SD["net.4.weight"].T + ENC_SD["net.4.bias"]


def field_type(col):
    if col in INPUT_LES:
        return "category"
    if col.strip().upper() == "AGE":
        return "number"
    return "binary"


# Form fields built from the saved feature list (no hard-coding)
FORM_FIELDS = [
    {
        "key":     f"f{i}",
        "name":    col,
        "label":   col.strip().replace("_", " ").title(),
        "type":    field_type(col),
        "options": list(INPUT_LES[col].classes_) if col in INPUT_LES else None,
    }
    for i, col in enumerate(FEATURES)
]


def grouped_fields():
    by_name = {f["name"].strip().upper(): f for f in FORM_FIELDS}
    out, used = [], set()
    for title, cols in GROUPS:
        items = [by_name[c] for c in cols if c in by_name]
        used.update(c for c in cols if c in by_name)
        if items:
            out.append({"title": title, "fields": items})
    rest = [f for k, f in by_name.items() if k not in used]
    if rest:
        out.append({"title": "Other", "fields": rest})
    return out


def preset_values():
    """PRESETS translated to form-field keys, for the JS quick-fill buttons."""
    out = {}
    for pid, p in PRESETS.items():
        vals = {}
        for f in FORM_FIELDS:
            key = f["name"].strip().upper()
            if f["type"] == "number":
                vals[f["key"]] = p["age"]
            elif f["type"] == "category":
                vals[f["key"]] = p["gender"]
            else:
                vals[f["key"]] = "YES" if key in p["yes"] else "NO"
        out[pid] = {"label": p["label"], "values": vals}
    return out


def to_value(col, value):
    """Convert one raw input value to the numeric coding used in training."""
    kind = field_type(col)
    if kind == "category":
        value = str(value).strip().upper()
        classes = [str(c).upper() for c in INPUT_LES[col].classes_]
        if value not in classes:
            raise ValueError(f"'{col.strip()}' must be one of {list(INPUT_LES[col].classes_)}")
        return int(INPUT_LES[col].transform([INPUT_LES[col].classes_[classes.index(value)]])[0])
    if kind == "number":
        return float(value)
    # binary symptom: accept YES/NO, 1/2, true/false
    v = str(value).strip().upper()
    if v in ("YES", "Y", "TRUE", str(YES)):
        return YES
    if v in ("NO", "N", "FALSE", str(NO), "0"):
        return NO
    raise ValueError(f"'{col.strip()}' must be YES/NO (or {YES}/{NO})")


def prepare(record):
    """record: dict {column name -> raw value}. Returns the 26-dim model input."""
    # tolerate keys with/without trailing spaces, any case
    norm = {str(k).strip().upper(): v for k, v in record.items()}
    row = {}
    for col in FEATURES:
        key = col.strip().upper()
        if key not in norm:
            raise ValueError(f"Missing field: '{col.strip()}'")
        row[col] = to_value(col, norm[key])

    frame    = pd.DataFrame([row], columns=FEATURES)
    x_scaled = scaler.transform(frame)
    x_sel    = x_scaled[:, TOP_IDX].astype(np.float32)
    z        = encoder_forward(x_sel)
    return np.hstack([x_sel, z])


def predict(record):
    """record: dict {column name -> raw value}. Returns (label, probability)."""
    x_final = prepare(record)
    prob = float(model.predict_proba(x_final)[0, 1])
    pred = int(prob >= 0.5)
    label = str(TARGET_LE.inverse_transform([pred])[0]) if TARGET_LE is not None else str(pred)
    return label, prob


# ---------------- Presentation helpers (read-only use of the model) ----------------
def pretty(col):
    return col.strip().replace("_", " ").title()


SELECTED   = [FEATURES[i] for i in TOP_IDX]
N_LATENT   = int(encoder_bundle["latent_dim"])
SELECTED_SET = {c.strip().upper() for c in SELECTED}


def explain(record):
    """Per-patient SHAP contributions (log-odds) from CatBoost's built-in TreeSHAP."""
    from catboost import Pool
    sv = model.get_feature_importance(Pool(prepare(record)), type="ShapValues")[0]
    norm = {str(k).strip().upper(): v for k, v in record.items()}
    rows = []
    for i, col in enumerate(SELECTED):
        raw = str(norm[col.strip().upper()]).strip()
        shown = {"M": "Male", "F": "Female"}.get(raw.upper(), raw.title())
        rows.append({"name": pretty(col), "value": shown, "shap": float(sv[i]), "latent": False})
    rows.sort(key=lambda r: abs(r["shap"]), reverse=True)
    peak = max(abs(r["shap"]) for r in rows) or 1.0
    for r in rows:
        r["pct"] = round(abs(r["shap"]) / peak * 100, 1)
    latent = float(sv[len(SELECTED):-1].sum())
    return {"rows": rows, "latent": latent, "base": float(sv[-1]), "total": float(sv.sum())}


def risk_band(prob):
    if prob < 0.35:
        return {"name": "Low risk", "cls": "low"}
    if prob < 0.65:
        return {"name": "Borderline risk", "cls": "mid"}
    return {"name": "High risk", "cls": "high"}


def global_importance():
    imp = model.get_feature_importance()
    names = [pretty(c) for c in SELECTED] + [f"z{i + 1}" for i in range(N_LATENT)]
    rows = [{"name": n, "value": float(v), "latent": i >= len(SELECTED)}
            for i, (n, v) in enumerate(zip(names, imp))]
    rows.sort(key=lambda r: r["value"], reverse=True)
    peak = rows[0]["value"] or 1.0
    for r in rows:
        r["pct"] = round(r["value"] / peak * 100, 1)
    latent_share = sum(r["value"] for r in rows if r["latent"])
    return {"rows": rows, "latent_share": latent_share, "selected_share": 100 - latent_share}


_params = model.get_params()
MODEL_INFO = {
    "n_records":   int(getattr(scaler, "n_samples_seen_", 0)),
    "n_raw":       len(FEATURES),
    "n_selected":  len(SELECTED),
    "n_latent":    N_LATENT,
    "n_final":     len(SELECTED) + N_LATENT,
    "selected":    [pretty(c) for c in SELECTED],
    "dropped":     [pretty(c) for c in FEATURES if c not in SELECTED],
    "encoder":     [int(encoder_bundle["input_dim"]),
                    *[int(ENC_SD[k].shape[0]) for k in ("net.0.weight", "net.2.weight", "net.4.weight")]],
    "catboost": [
        ("Iterations (max)",   _params.get("iterations")),
        ("Trees after early stop", model.tree_count_),
        ("Learning rate",      _params.get("learning_rate")),
        ("Tree depth",         _params.get("depth")),
        ("Loss function",      _params.get("loss_function")),
        ("Eval metric",        _params.get("eval_metric")),
        ("Early stopping",     f'{_params.get("early_stopping_rounds")} rounds'),
        ("Random seed",        _params.get("random_seed")),
    ],
    "importance":  global_importance(),
}

# Field grouping for the form (unknown columns fall into "Other")
GROUPS = [
    ("Demographics",          ["GENDER", "AGE"]),
    ("Lifestyle & Habits",    ["SMOKING", "ALCOHOL CONSUMING", "PEER_PRESSURE"]),
    ("Clinical Symptoms",     ["COUGHING", "CHEST PAIN", "SHORTNESS OF BREATH", "WHEEZING",
                               "SWALLOWING DIFFICULTY", "YELLOW_FINGERS", "FATIGUE"]),
    ("History & Wellbeing",   ["CHRONIC DISEASE", "ALLERGY", "ANXIETY"]),
]

# Demo patients for live presentation (raw values keyed by column name)
PRESETS = {
    "low":  {"label": "Low-risk patient", "age": 35, "gender": "F", "yes": []},
    "mid":  {"label": "Borderline patient", "age": 58, "gender": "M",
             "yes": ["SMOKING", "COUGHING", "CHEST PAIN"]},
    "high": {"label": "High-risk patient", "age": 67, "gender": "M",
             "yes": ["SMOKING", "YELLOW_FINGERS", "COUGHING", "CHEST PAIN", "SWALLOWING DIFFICULTY",
                     "WHEEZING", "SHORTNESS OF BREATH", "FATIGUE", "ALCOHOL CONSUMING",
                     "CHRONIC DISEASE"]},
}


app = Flask(__name__)


def page(template, name, **ctx):
    return render_template(template, page=name, metrics=metrics, info=MODEL_INFO, **ctx)


@app.route("/")
def home():
    return page("home.html", "home")


@app.route("/methodology")
def methodology():
    return page("methodology.html", "methodology")


@app.route("/performance")
def performance():
    return page("performance.html", "performance")


@app.route("/predict", methods=["GET", "POST"], endpoint="predict")
def predict_page():
    error, values = None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        edit = values.pop("_edit", None)   # "Edit inputs" from the result page
        if not edit:
            try:
                record = {f["name"]: values.get(f["key"], "") for f in FORM_FIELDS}
                label, prob = predict(record)
                result = {"label": label, "prob": round(prob * 100, 2),
                          "positive": label.strip().upper() in ("YES", "1"),
                          "band": risk_band(prob), "explain": explain(record)}
                return page("result.html", "predict", result=result, values=values)
            except Exception as e:
                error = str(e)
    return page("predict.html", "predict", groups=grouped_fields(), error=error, values=values,
                selected_set=SELECTED_SET, presets=preset_values())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API. Body: one dict or a list of dicts with the column names."""
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Send JSON body"}), 400
    records = payload if isinstance(payload, list) else [payload]
    try:
        out = []
        for r in records:
            label, prob = predict(r)
            out.append({"prediction": label, "probability": round(prob, 4)})
        return jsonify(out if isinstance(payload, list) else out[0])
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "features": [f.strip() for f in FEATURES]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
