import os
import streamlit as st
import pandas as pd
import pickle

st.title("ESG Score Calculator and Rating")
csv_file_path = "merged.csv"

@st.cache_resource
def load_model_and_scaler():
    try:
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        with open("tpot_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        return scaler, model
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)

if not os.path.exists(csv_file_path):
    st.error(f"CSV file not found at: {csv_file_path}. Please verify the path.")
    st.stop()

df = load_csv(csv_file_path)
scaler, model = load_model_and_scaler()

column_mapping = {
    "COMPANY_NAME": "Company",
    "IVA_INDUSTRY": "Type",
    "ENVIRONMENTAL_PILLAR_SCORE": "Environmental Score",
    "SOCIAL_PILLAR_SCORE": "Social Score",
    "GOVERNANCE_PILLAR_SCORE": "Governance Score",
}
df.rename(columns=column_mapping, inplace=True)

if "Type" not in df.columns:
    df["Type"] = "Unknown"

st.markdown("### Search for a Company")
selected_company = st.text_input("Enter Company Name:", placeholder="Type company name here")

if selected_company:
    matching_companies = df[df["Company"].str.contains(selected_company, na=False, case=False)]
    if not matching_companies.empty:
        st.success(f"Company '{selected_company}' found in the dataset.")
        st.dataframe(matching_companies)

        visualization_url = "https://studious-memory-q7ppxr6vgqr39qj5-8501.app.github.dev/Visualization"
        st.markdown(f"[Go to Visualization]({visualization_url})", unsafe_allow_html=True)
    else:
        st.warning(f"Company '{selected_company}' not found in the dataset.")

        st.markdown("### Add New Company")
        company_type = st.selectbox("Select Company Type:", options=df["Type"].unique())
        environmental_score = st.slider("Environmental Score (0-10):", 0, 10, 10)
        social_score = st.slider("Social Score (0-10):", 0, 10, 10)
        governance_score = st.slider("Governance Score (0-10):", 0, 10, 10)
        Industry_Adjusted_Score = st.slider("Industry Adjusted Score (0-10):",0,10,10)
        Carbon_Emission_Score = st.slider("Carbon Emission Score (0-10):",0,10,10)

        if scaler and model:
            # Prepare input for the model
            input_data = [[environmental_score, governance_score, social_score]]
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]

            st.success(f"The predicted ESG rating for '{selected_company}' is: {prediction}")
        else:
            st.error("Model and scaler not properly loaded. Cannot predict ESG rating.")

        if st.button("Save New Company"):
            new_data = {
                "Company": selected_company,
                "Type": company_type,
                "Environmental Score": environmental_score,
                "Social Score": social_score,
                "Governance Score": governance_score,
            }
            new_df = pd.DataFrame([new_data])
            new_df.to_csv(csv_file_path, mode="a", header=False, index=False)
            st.success(f"New company '{selected_company}' added successfully!")
