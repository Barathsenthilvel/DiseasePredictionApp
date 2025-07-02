import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import io
import base64
from collections import Counter

from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load datasets
df_main = pd.read_csv("dataset.csv")
df_description = pd.read_csv("symptom_Description.csv")
df_precaution = pd.read_csv("symptom_precaution.csv")
df_severity = pd.read_csv("Symptom-severity.csv")

all_symptoms = sorted(set(df_severity['Symptom'].str.strip().str.lower()))

# Prepare symptoms_series for top symptoms plot
symptoms_data = df_main.iloc[:, 1:]  # Assuming symptoms start from 2nd column
all_symptoms_list = symptoms_data.values.flatten()
symptoms_series = pd.Series(all_symptoms_list).dropna().astype(str).str.lower()

# Encode symptoms to binary vector
def encode_symptoms(symptom_list):
    symptoms_vector = [0] * len(all_symptoms)
    for symptom in symptom_list:
        if pd.isna(symptom):
            continue
        symptom = str(symptom).strip().lower()
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            symptoms_vector[index] = 1
    return symptoms_vector

# Prepare training data
X = []
y = []

for _, row in df_main.iterrows():
    symptoms = row[1:].fillna('').astype(str).tolist()
    X.append(encode_symptoms(symptoms))
    y.append(row['Disease'])

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = DecisionTreeClassifier()
model.fit(X, y_encoded)

# Function to generate bar chart of user input symptoms

def generate_symptom_plot(symptoms_list):
    counts = Counter([s.lower() for s in symptoms_list])
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(8, 4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.title("Your Input Symptoms Frequency")
    plt.xlabel("Symptoms")
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, yval, ha='center')

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_data

# Function to show top 10 most common symptoms
def plot_top_symptoms():
    top_symptoms = symptoms_series.value_counts().head(10)
    plt.figure(figsize=(10, 6))
    colors = cm.tab20(np.arange(len(top_symptoms)))
    bars = plt.bar(top_symptoms.index, top_symptoms.values, color=colors)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, int(yval), ha='center', va='bottom')

    plt.title('Top 10 Most Common Symptoms', fontsize=16, fontweight='bold')
    plt.xlabel('Symptoms', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_data = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_data

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    description = None
    precautions = []
    plot_url = None

    if request.method == "POST":
        symptoms = request.form.get("symptoms")
        if symptoms:
            user_symptoms = [s.strip() for s in symptoms.split(",")]
            user_input = encode_symptoms(user_symptoms)
            pred_code = model.predict([user_input])[0]
            prediction = le.inverse_transform([pred_code])[0]
            plot_url = generate_symptom_plot(user_symptoms)

            desc_row = df_description[df_description['Disease'] == prediction]
            if not desc_row.empty:
                description = desc_row['Description'].values[0]

            prec_row = df_precaution[df_precaution['Disease'] == prediction]
            if not prec_row.empty:
                for i in range(1, 5):
                    col = f'Precaution_{i}'
                    if col in prec_row.columns and pd.notna(prec_row[col].values[0]):
                        precautions.append(prec_row[col].values[0])
    else:
        plot_url = plot_top_symptoms()

    return render_template("index.html", prediction=prediction, description=description,
                           precautions=precautions, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
