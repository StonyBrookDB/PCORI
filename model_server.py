from flask import Flask, request, jsonify
import pandas as pd
import pymysql
import joblib
import requests
from flask_cors import CORS


app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

CORS(app)

def get_risk_level(score):
    if score > 0.7:
        return "High", "Immediate attention needed."
    elif score > 0.4:
        return "Medium", "Monitor regularly."
    else:
        return "Low", "Stable condition."

def generate_llm_reasoning(patient_id, risk_score, risk_level, top_features):
    features_text = "\n".join(
        [f"{f['ranking']}: score {f['score']}" for f in top_features]
    )
    prompt = f"""
Patient ID: {patient_id}
Risk Score: {risk_score}
Risk Level: {risk_level}
Top Risk-Contributing Features:
{features_text}

Explain in 5 lines why the patient might be at {risk_level} risk level based on these features.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=30
        )
        result = response.json()
        return result.get("response", "LLM reasoning unavailable.")
    except Exception as e:
        return f"Error getting reasoning: {e}"

@app.route("/risk_score", methods=["POST"])
def get_risk_score():
    try:
        patient_id = request.json.get("patient_id")

        # Connect to MySQL
        conn = pymysql.connect(
            host="localhost",
            user="root",
            password="root",
            database="pcori_dashboard",
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            query = """
                SELECT score, ranking
                FROM t_topfeature
                WHERE patient_id = %s
                ORDER BY ranking ASC
                LIMIT 10
            """
            cursor.execute(query, (patient_id,))
            rows = cursor.fetchall()

        if not rows:
            return jsonify({"error": "No data found for patient"}), 404

        df = pd.DataFrame(rows)
        prediction_prob = model.predict_proba(df[["score", "ranking"]])[:, 1][0]
        risk_level, explanation = get_risk_level(prediction_prob)

        top_features = [
            {"ranking": row["ranking"], "score": row["score"]}
            for row in rows[:5]
        ]

        llm_reasoning = generate_llm_reasoning(patient_id, prediction_prob, risk_level, top_features)

        return jsonify({
            "risk_score": round(float(prediction_prob), 2),
            "risk_level": risk_level,
            "explanation": explanation,
            "top_features": top_features,
            "llm_reasoning": llm_reasoning
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


    
@app.route("/all_patient_details", methods=["GET"])
def get_all_patient_details():
    try:
        # Connect to MySQL
        conn = pymysql.connect(
            host="127.0.0.1",    # localhost
            port=3307,           # forwarded port
            user="sankeer",      # your database username
            password="pwdsankeer",  # your database password
            database="pcori_dashboard", # database name
            cursorclass=pymysql.cursors.DictCursor
        )

        with conn.cursor() as cursor:
            query = """
                SELECT tp.patient_id, tp.patient_sk, tp.race, tp.gender, tp.marital_status,
                       tm.encounter_id, tm.mme_score, tm.encounter_date
                FROM t_patient tp
                LEFT JOIN t_MME tm ON tp.patient_id = tm.patient_id
            """
            cursor.execute(query)
            all_patients = cursor.fetchall()

        if not all_patients:
            return jsonify({"error": "No patient data found"}), 404

        clean_data = []
        for row in all_patients:
            clean_data.append({
                "patient_id": str(row["patient_id"]),
                "patient_sk": str(row["patient_sk"]),
                "race": row["race"],
                "gender": row["gender"],
                "marital_status": row["marital_status"],
                "encounter_id": row["encounter_id"],
                "mme_score": row["mme_score"],
                "encounter_date": row["encounter_date"].strftime("%Y-%m-%d") if row["encounter_date"] else None
            })


        return jsonify(clean_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(port=5006)
