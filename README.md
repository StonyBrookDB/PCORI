# Patient Risk Score Dashboard using Tableau, TabPy & MySQL

This project visualizes patient risk scores (e.g., OUD/OD prediction) using mock data from a MySQL database. Tableau connects to Python (TabPy) running on a remote server to get real-time model predictions.


# What It Does

- Loads patient feature data from MySQL
- Trains a Random Forest model on the features
- Runs a Python Flask server to provide predictions
- Connects Tableau to the model via TabPy
- Displays risk score, level (Low/Medium/High), and key features


# Project Files

- `train_model.py`: Trains and saves the ML model
- `model_server.py`: Flask server that fetches from MySQL and returns predictions
- `random_forest_model.pkl`: Saved trained model
- `tabpy_env/`: Python virtual environment (not required to upload)


# Setup

### On Remote Server (`bmidb0`)
bash
# SSH into remote
ssh -p 130 username@bmidb0.cs.stonybrook.edu

# Activate environment
cd ~/pcori_sample_project
source tabpy_env/bin/activate

# Train the model
python train_model.py

# Start TabPy model server
python model_server.py

# On Local Machine (for Tableau)
bash
# Set up SSH tunnel to TabPy
ssh -N -L 9004:localhost:9004 -p 130 reddy@bmidb0.cs.stonybrook.edu

---

# Tableau Setup

1. Connect Tableau to remote MySQL DB (`pcori_dashboard.t_topfeature`) (optional) or use the excel data and upload the file.
2. Go to `Help → Settings → Manage Analytics Extension Connection`
3. Connect to `localhost:9004` (TabPy)
4. Use calculated field with:

python

Risk Explanation:-


SCRIPT_STR(
"
import requests
r = requests.post(
    'http://localhost:5006/risk_score',
    json={'patient_id': int(_arg1[0])}
)
return [r.json()['explanation']]
", 
ATTR([Patient])
)


Risk Level:-

SCRIPT_STR(
"
import requests
r = requests.post(
    'http://localhost:5006/risk_score',
    json={'patient_id': int(_arg1[0])}
)
return [r.json()['risk_level']]
", 
ATTR([Patient])
)


Risk Score:- 

SCRIPT_REAL(
"
import requests
r = requests.post(
    'http://localhost:5006/risk_score',
    json={'patient_id': int(_arg1[0])}
)
return [r.json()['risk_score']]
", 
ATTR([Patient])
)


Top Feature:- 

SCRIPT_REAL(
"
import requests
r = requests.post(
    'http://localhost:5006/risk_score',
    json={'patient_id': int(_arg1[0])}
)
return [r.json()['top_features'][0]['score']]
", 
ATTR([Patient])
)


# Output

- Risk Score (0–1)
- Risk Level: Low (<0.4), Medium (0.4–0.7), High (>0.7)
- Top Features (ranked by importance)


# Notes

- Entire pipeline runs remotely on bmidb0 server.
- Tableau fetches results via TabPy API.