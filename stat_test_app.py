import streamlit as st
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load("stat_test_selector_xgb.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, vectorizer, encoder

model, vectorizer, encoder = load_artifacts()

st.set_page_config(page_title="Statistical Test Selector", layout="centered")

st.title("🔬 Statistical Test Selector")
st.write("Введите описание биомедицинской статистической задачи, и модель подберёт корректный статистический тест.")

user_input = st.text_area(
    "Описание задачи:",
    placeholder="Например: сравнение двух независимых групп по уровню глюкозы..."
)

if st.button("Предсказать тест"):
    if user_input.strip() == "":
        st.error("Введите описание задачи!")
    else:
        X = vectorizer.transform([user_input])
        pred = model.predict(X)[0]
        predicted_test = encoder.inverse_transform([pred])[0]

        st.success(f"Рекомендуемый статистический тест: **{predicted_test}**")
