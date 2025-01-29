import os
import streamlit as st
import pandas as pd
import pickle


csv_file_path = "merged_output.csv"
rating_mapping = [
    {"Rating": "A", "Min": 4.1, "Max": 4.8},
    {"Rating": "AA", "Min": 4.9, "Max": 5.5},
    {"Rating": "AAA", "Min": 5.6, "Max": 100},
    {"Rating": "B", "Min": 2.4, "Max": 2.9},
    {"Rating": "BB", "Min": 3.0, "Max": 3.6},
    {"Rating": "BBB", "Min": 3.7, "Max": 4.0},
    {"Rating": "CCC", "Min": 0, "Max": 2.3},
]

column_mapping = {
    "COMPANY_NAME": "Company",
    "IVA_INDUSTRY": "Type",
    "ENVIRONMENTAL_PILLAR_SCORE": "Environmental Score",
    "SOCIAL_PILLAR_SCORE": "Social Score",
    "GOVERNANCE_PILLAR_SCORE": "Governance Score",
    "WEIGHTED_AVERAGE_SCORE": "ESG Score"
}


def classify_rating(score):
    for rating in rating_mapping:
        if rating["Min"] <= score <= rating["Max"]:
            return rating["Rating"]
    return "No Rating"

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

df = load_csv(csv_file_path)
scaler, model = load_model_and_scaler()
df.rename(columns=column_mapping, inplace=True)


st.title("ESG Score Calculator and Rating")
st.markdown("### Search for a Company")
selected_company = st.text_input("Enter Company Name and Press Enter:", placeholder="Type company name here")

if selected_company:

    matching_companies = df[df["Company"].str.contains(selected_company, na=False, case=False)]
    if not matching_companies.empty:
        st.success(f"Company '{selected_company}' found in the dataset.")
        st.dataframe(matching_companies)
    else:
        st.warning(f"Company '{selected_company}' not found in the dataset.")

        st.markdown("### Add New Company")
        company_type = st.selectbox("Select Company Type:", options=df["Type"].unique())
        environmental_score = st.slider("Environmental Score (0.00-10.00):", 0.0, 10.0, 10.0, step=0.01)
        social_score = st.slider("Social Score (0.00-10.00):", 0.0, 10.0, 10.0, step=0.01)
        governance_score = st.slider("Governance Score (0.00-10.00):", 0.0, 10.0, 10.0, step=0.01)

        if st.button("Save New Company"):
            if scaler and model:
                input_data = [[environmental_score, governance_score, social_score]]
                scaled_data = scaler.transform(input_data)
                prediction = model.predict(scaled_data)[0]
                predicted_rating = classify_rating(prediction)

                st.write(
                        f"""
                            <div style="
                                background-color: rgb(16 14 14 / 62%);
                                color: #ffffff;
                                padding: 15px;
                                border-radius: 10px;
                                border: 2px solid #ff4b4b;
                                box-shadow: 2px 2px 10px rgba(255,255,255,0.2);
                                margin: 20px;
                            ">
                                <h3 style="color: #ff4b4b;">🎯 Prediction Result</h3>
                                <p><strong>Company Name:</strong> <span style="color: #ff4b4b;">{selected_company}</span></p>
                                <p><strong>Predicted ESG Rating:</strong> <span style="color: #ff4b4b;">{predicted_rating}</span></p>
                                <p><strong>ESG Score:</strong> <span style="color: #ff5722;">{prediction:.2f}</span></p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.error("Model and scaler not properly loaded. Cannot predict ESG rating.")
            
            new_data = {
                "IVA_COMPANY_RATING": predicted_rating,
                "COMPANY_NAME": selected_company,
                "ENVIRONMENTAL_PILLAR_SCORE": environmental_score,
                "GOVERNANCE_PILLAR_SCORE": governance_score,
                "SOCIAL_PILLAR_SCORE": social_score,
                "IVA_INDUSTRY": company_type,
                "WEIGHTED_AVERAGE_SCORE": prediction
            }
            new_df = pd.DataFrame([new_data])
            new_df.to_csv(csv_file_path, mode="a", header=False, index=False)
            st.cache_data.clear()
            st.success(f"New company '{selected_company}' added successfully!")

        # if scaler and model:
        #     # Predict ESG Score and Rating
        #     input_data = [[environmental_score, governance_score, social_score]]
        #     scaled_data = scaler.transform(input_data)
        #     prediction = model.predict(scaled_data)[0]
        #     predicted_rating = classify_rating(prediction)

        #     st.write(
        #         f"""
        #             <div style="
        #                 background-color: rgb(16 14 14 / 62%);
        #                 color: #ffffff;
        #                 padding: 15px;
        #                 border-radius: 10px;
        #                 border: 2px solid #ff4b4b;
        #                 box-shadow: 2px 2px 10px rgba(255,255,255,0.2);
        #                 margin: 20px;
        #             ">
        #                 <h3 style="color: #ff4b4b;">🎯 Prediction Result</h3>
        #                 <p><strong>Company Name:</strong> <span style="color: #ff4b4b;">{selected_company}</span></p>
        #                 <p><strong>Predicted ESG Rating:</strong> <span style="color: #ff4b4b;">{predicted_rating}</span></p>
        #                 <p><strong>ESG Score:</strong> <span style="color: #ff5722;">{prediction:.2f}</span></p>
        #             </div>
        #         """,
        #         unsafe_allow_html=True
        #     )
        # else:
        #     st.error("Model and scaler not properly loaded. Cannot predict ESG rating.")




st.cache_data.clear()
# 281102