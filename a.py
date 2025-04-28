import streamlit as st
import pandas as pd
import joblib
import io

def load_model():
    return joblib.load('randomforest_model.pkl'), joblib.load('standard_scaler.pkl')

model, scaler = load_model()

st.set_page_config(page_title="EEG Emotion Predictor", layout="centered")
st.title("🧠 EEG Emotion Prediction App")
st.markdown("Upload EEG data and get emotion predictions instantly.")

uploaded_file = st.file_uploader("Upload your EEG CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Uploaded Data Preview")
        st.dataframe(df)

        st.subheader("Data Information")
        col1, col2 = st.columns(2)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])

        required_features = scaler.mean_.shape[0]
        if df.shape[1] != required_features:
            st.error(f"Expected {required_features} features, but got {df.shape[1]}. Please check your file.")
        else:
            with st.spinner('Predicting...'):
                X_scaled = scaler.transform(df)
                preds = model.predict(X_scaled)

            st.success("Prediction completed successfully!")

            pred_df = pd.DataFrame({
                "Sample ID": range(1, len(preds)+1),
                "Predicted Emotion": preds
            })

            st.subheader("Predictions")
            st.dataframe(pred_df)

            csv_buffer = io.StringIO()
            pred_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="Download Predictions as CSV",
                data=csv_data,
                file_name='predicted_emotions.csv',
                mime='text/csv'
            )



    except Exception as e:
        st.error(f"An error occurred: {e}")
