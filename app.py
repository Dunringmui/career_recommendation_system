import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model and encoders
model = joblib.load("best_model.pkl")
career_encoder = joblib.load("career_label_encoder.pkl")
model_input_columns = joblib.load("model_input_columns.pkl")

categorical_cols = [
    "Stream", "Activity", "Interest1", "Interest2", "Skill1", "Skill2",
    "Subject1", "Subject2", "PreferredEnv", "StudyStyle"
]
encoders = {col: joblib.load(f"{col.lower()}_encoder.pkl") for col in categorical_cols}

# Streamlit UI
st.title("Career Recommendation System")

# Collect input for categorical fields
raw_input = {}
st.markdown("### Basic Profile")
for col in categorical_cols:
    options = encoders[col].classes_.tolist()
    raw_input[col] = st.selectbox(f"Select {col}", options)

# Initialize quiz scores
num_score = 0
logic_score = 0
eng_score = 0
person_score = 0

# Numerical Ability
st.markdown("**Numerical Ability**")
numerical_qs = [
    ("What is the value of 18² - 12²?", ["144", "324", "180", "216"], "180"),
    ("If 12 pens cost ₹144, what is the cost of 5 pens?", ["₹50", "₹60", "₹65", "₹70"], "₹60"),
    ("What is 2.5% of 800?", ["10", "20", "25", "30"], "20"),
    ("Solve: 1/4 of 1/2 of 400", ["25", "50", "100", "200"], "50"),
    ("A car travels 150 km in 3 hours. What is its average speed?", ["30 km/h", "40 km/h", "50 km/h", "60 km/h"], "50 km/h"),
    ("What is the next prime number after 17?", ["18", "19", "20", "21"], "19"),
    ("Find the LCM of 12 and 15.", ["60", "30", "45", "75"], "60"),
    ("What is the square root of 625?", ["15", "20", "25", "30"], "25"),
    ("If a = 4 and b = 2, then (a² + b²) = ?", ["16", "20", "18", "12"], "20"),
    ("Simplify: (3/5) ÷ (9/10)", ["2/3", "1/3", "3/2", "5/6"], "2/3")
]
for i, (q, options, correct) in enumerate(numerical_qs):
    ans = st.radio(f"Q{i+1}: {q}", options, key=f"num_{i}")
    if ans == correct:
        num_score += 1

# Logical Reasoning
st.markdown("**Logical Reasoning**")
logic_qs = [
    ("Which number completes the series: 5, 10, 20, 40, ___", ["50", "60", "70", "80"], "80"),
    ("Find the odd one out: Apple, Orange, Banana, Carrot", ["Apple", "Banana", "Orange", "Carrot"], "Carrot"),
    ("What comes next: A, C, E, G, ___", ["H", "I", "J", "K"], "I"),
    ("If John is taller than Sam and Sam is taller than Mike, who is shortest?", ["Mike", "Sam", "John", "Can’t say"], "Mike"),
    ("Which number is missing: 3, 6, 9, __, 15", ["11", "12", "13", "14"], "12"),
    ("If MONDAY is coded as NPOEBZ, what is the code for FRIDAY?", ["GSJEBZ", "GRIEBZ", "GSJDBZ", "GRIHBZ"], "GSJEBZ"),
    ("What comes next: 100, 96, 88, 76, ___", ["70", "64", "60", "66"], "60"),
    ("If all Bloops are Razzies and all Razzies are Lazzies, then all Bloops are:", ["Razzies", "Lazzies", "None", "Cannot say"], "Lazzies"),
    ("If ‘CAT’ is 3 + 1 + 20 = 24, what is ‘DOG’?", ["26", "30", "32", "34"], "26"),
    ("Complete the analogy: Book : Reading :: Fork : ___", ["Drawing", "Stirring", "Writing", "Eating"], "Eating")
]
for i, (q, options, correct) in enumerate(logic_qs):
    ans = st.radio(f"Q{i+1}: {q}", options, key=f"log_{i}")
    if ans == correct:
        logic_score += 1

# English Grammar
st.markdown("**English Grammar**")
english_qs = [
    ("Identify the correct sentence.", ["He go to school.", "He goes to school.", "He going school."], "He goes to school."),
    ("Choose the correct verb: She ___ tea every morning.", ["drink", "drinks", "drinking", "drank"], "drinks"),
    ("Plural of 'child' is:", ["childs", "children", "childes", "childs'"], "children"),
    ("Choose the opposite of 'generous':", ["kind", "rude", "selfish", "honest"], "selfish"),
    ("Fill in the blank: I have lived here ___ 5 years.", ["since", "for", "from", "in"], "for"),
    ("What is the synonym of 'happy'?", ["sad", "joyful", "angry", "tired"], "joyful"),
    ("Identify the adjective: The quick brown fox jumped.", ["quick", "fox", "jumped", "brown"], "quick"),
    ("Change to past tense: She writes a letter.", ["She writing", "She wrote", "She written", "She write"], "She wrote"),
    ("What is the noun in: The sun shines brightly.", ["sun", "shines", "brightly", "The"], "sun"),
    ("Which sentence is correct?", ["We was late.", "They is here.", "She is fine."], "She is fine.")
]
for i, (q, options, correct) in enumerate(english_qs):
    ans = st.radio(f"Q{i+1}: {q}", options, key=f"eng_{i}")
    if ans == correct:
        eng_score += 1

# Personality Assessment
st.markdown("**Personality Assessment**")
personality_qs = [
    "I enjoy working with people and helping them.",
    "I prefer working alone rather than in a group.",
    "I like trying new things even if they seem difficult.",
    "I find it easy to manage time and stay organized.",
    "I stay calm even in stressful situations.",
    "I feel confident when speaking in public.",
    "I like solving puzzles and logical problems.",
    "I enjoy taking responsibility and leading others.",
    "I pay attention to details in everything I do.",
    "I prefer following clear rules rather than taking risks."
]
for i, q in enumerate(personality_qs):
    person_score += st.slider(f"Q{i+1}: {q}", 1, 5, 3, key=f"p{i}")

# Predict
if st.button("Recommend My Career"):
    encoded_input = {}
    for col in categorical_cols:
        encoded_input[col] = encoders[col].transform([raw_input[col]])[0]

    encoded_input.update({
        "Numerical": num_score,
        "Logical": logic_score,
        "English": eng_score,
        "Personality": person_score,
        "Academic_Aptitude": (num_score + logic_score) / 2,
        "Language_Aptitude": eng_score
    })

    input_df = pd.DataFrame([encoded_input])
    for col in model_input_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_input_columns]

    proba = model.predict_proba(input_df)[0]
    top_3 = np.argsort(proba)[::-1][:3]
    st.subheader("Top 3 Career Recommendations:")
    for idx in top_3:
        st.write(f"{career_encoder.inverse_transform([idx])[0]} - {proba[idx]*100:.2f}% confidence")
