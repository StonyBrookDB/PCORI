'''
This file is present at "/home/kavya/MODELS/Llama-3.1-8B-Instruct". 
This code provisions a Flask Server to respond to Prompt Requests
'''

from transformers import pipeline
import torch
import json
from flask import Flask, request, jsonify



model_id = "/home/kavya/MODELS/Llama-3.1-8B-Instruct"

pipe = pipeline(
    "text-generation",
    model = model_id,
    model_kwargs = {"torch_dtype": torch.bfloat16},
    device="cuda" #GPU
    #device = -1 #cpu-executable
)


def LLM_response( reqq ):
    op =  pipe(
        [ {"role": "user", "content": reqq } ],
        max_new_tokens = 256,
        do_sample= False 
    )
    print("Request: " + reqq)
    a = op[0]["generated_text"][-1]["content"]
    print("Response: " + a)
    return a

# Test LLM with Mock data
def test_LlamaModel():
    patient_id = "1"
    risk_score = 0.8
    risk_level = "High Risk"
    features_text = json.dumps( {
    "t_topfeature": [
        {
            "ranking" : 1,
            "score" : 91.5
        },
        {
            "ranking" : 2,
            "score" : 89.7
        },
        {
            "ranking" : 3,
            "score" : 90.3
        },
        {
            "ranking" : 4,
            "score" : 88.8
        },
        {
            "ranking" : 5,
            "score" : 87.6
        },
        {
            "ranking" : 6,
            "score" : 86.4
        },
        {
            "ranking" : 7,
            "score" : 85.2
        },
        {
            "ranking" : 8,
            "score" : 84.0
        },
        {
            "ranking" : 9,
            "score" : 83.5
        },
        {
            "ranking" : 10,
            "score" : 82.1
        }
    ]}
    )



    prompt = f"""
    Patient ID: {patient_id}
    Risk Score: {risk_score}
    Risk Level: {risk_level}
    Top Risk-Contributing Features:
    {features_text}

    Explain in 5 lines why the patient might be at {risk_level} risk level based on these features.
        """


    print("\n\n\n\n\n" + LLM_response(prompt))




app = Flask(__name__)

@app.route("/llama", methods=["POST"])
def get_llama_model_reasoning():
    try:
        print(request.json)
        req = request.json.get("content")
        resp = LLM_response(req)

        return jsonify({
            "llm_reasoning" : resp
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    #test_LlamaModel()
    app.run(port=8765)

