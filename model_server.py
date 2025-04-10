from flask import Flask, request, jsonify
import pandas as pd
import pymysql
import joblib

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

def get_risk_level(score):
    if score > 0.7:
        return "High", "Immediate attention needed."
    elif score > 0.4:
        return "Medium", "Monitor regularly."
    else:
        return "Low", "Stable condition."

@app.route("/risk_score", methods=["POST"])
def get_risk_score():
    try:
        patient_id = request.json.get("patient_id")

        # Connect to MySQL
        conn = pymysql.connect(
            host="localhost",
            user="reddy",
            password="pwdreddy",
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

        return jsonify({
            "risk_score": round(float(prediction_prob), 2),
            "risk_level": risk_level,
            "explanation": explanation,
            "top_features": top_features
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5006)
