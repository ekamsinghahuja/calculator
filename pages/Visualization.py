import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

csv_file_path = "merged_output.csv"
st.title("Enhanced ESG Ratings Visualization")
st.title("Company ESG Data Viewer")



def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at path: {file_path}")
        return None


df = load_csv(csv_file_path)

if df is not None:
    required_columns = [
        "COMPANY_NAME",
        "IVA_COMPANY_RATING",
        "ENVIRONMENTAL_PILLAR_SCORE",
        "GOVERNANCE_PILLAR_SCORE",
        "SOCIAL_PILLAR_SCORE",
        "IVA_INDUSTRY",
        "WEIGHTED_AVERAGE_SCORE"
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"The following required columns are missing from the CSV: {', '.join(missing_columns)}")
    else:
        df_filtered = df[required_columns]
        column_mapping = {
            "IVA_COMPANY_RATING": "IVA COMPANY RATING",
            "COMPANY_NAME": "COMPANY NAME",
            "ENVIRONMENTAL_PILLAR_SCORE": "ENVIRONMENTAL SCORE",
            "GOVERNANCE_PILLAR_SCORE": "GOVERNANCE SCORE",
            "SOCIAL_PILLAR_SCORE": "SOCIAL SCORE",
            "IVA_INDUSTRY": "INDUSTRY TYPE",
            "WEIGHTED_AVERAGE_SCORE":"ESG Score"
        }
        df_filtered.rename(columns=column_mapping, inplace=True)
        st.markdown("### ESG Data Table")
        st.dataframe(df_filtered, use_container_width=True)

else:
    st.error("Failed to load the CSV file.")

def load_csv(file_path):
    return pd.read_csv(file_path)

if not os.path.exists(csv_file_path):
    st.error(f"CSV file not found at: {csv_file_path}. Please verify the path.")
    st.stop()

df = load_csv(csv_file_path)
column_mapping = {
    "COMPANY_NAME": "Company",
    "IVA_INDUSTRY": "Type",
    "ENVIRONMENTAL_PILLAR_SCORE": "Environmental Score",
    "SOCIAL_PILLAR_SCORE": "Social Score",
    "GOVERNANCE_PILLAR_SCORE": "Governance Score",
    "WEIGHTED_AVERAGE_SCORE":"ESG Score"
}
df.rename(columns=column_mapping, inplace=True)



if "Type" not in df.columns:
    df["Type"] = "Unknown"

# Unique types from the dataset
types = df["Type"].unique().tolist()

# Global filter for the entire dataset
st.sidebar.markdown("### Global Filters")
global_company_filter = st.sidebar.multiselect("Filter by Company", options=df["Company"].unique(), default=[])
global_type_filter = st.sidebar.multiselect("Filter by Type", options=df["Type"].unique(), default=[])

if global_company_filter:
    df = df[df["Company"].isin(global_company_filter)]
if global_type_filter:
    df = df[df["Type"].isin(global_type_filter)]

# User input for company details
st.markdown("### Enter Company Details")
company_name = st.text_input("Search Company Name", value="", placeholder="Enter the company name")

# Dynamically suggest company names in dropdown
matching_companies = (
    df[df["Company"].str.contains(company_name, na=False, case=False)] if company_name else pd.DataFrame()
)
if not matching_companies.empty:
    suggestions = matching_companies["Company"].unique()
    company_name = st.selectbox("Matching Companies:", suggestions, key="company_selector")
    st.success(f"Company '{company_name}' selected.")
    st.session_state["selected_company"] = company_name
    st.session_state["selected_type"] = df.loc[df["Company"] == company_name, "Type"].values[0]
else:
    if company_name:
        st.warning("No matching company found. You can add it below.")

# Add navigation links with improved styling
st.markdown("### Graph Navigation")
st.markdown("""
<div style= padding:10px; border-radius:5px;">
    <ul style="list-style-type:none; padding-left:0;">
        <li><a href="#environmental-score-comparison" style="text-decoration:none; color:#007bff;"> Environmental Score Comparison</a></li>
        <li><a href="#social-score-comparison" style="text-decoration:none; color:#007bff;"> Social Score Comparison</a></li>
        <li><a href="#governance-score-comparison" style="text-decoration:none; color:#007bff;">Governance Score Comparison</a></li>
        <li><a href="#3d-esg-scatter-plot" style="text-decoration:none; color:#007bff;">3D ESG Scatter Plot</a></li>
        <li><a href="#radar-chart" style="text-decoration:none; color:#007bff;">Radar Chart</a></li>
        <li><a href="#aa" style="text-decoration:none; color:#007bff;">Enhanced  Environmental vs Social Scores</a></li>
        <li><a href="#ab" style="text-decoration:none; color:#007bff;">Enhanced Environmental vs Governance Scores</a></li>
        <li><a href="#ac" style="text-decoration:none; color:#007bff;">Enhanced Social vs Governance Scores</a></li>
        <li><a href="#ad" style="text-decoration:none; color:#007bff;">Stacked Bar Chart of ESG Scores by Company</a></li>
        <li><a href="#ae" style="text-decoration:none; color:#007bff;">Histogram of ESG Scores</a></li>
        <li><a href="#af" style="text-decoration:none; color:#007bff;">ESG Score vs Environmental Score</a></li>
        <li><a href="#ag" style="text-decoration:none; color:#007bff;">ESG Score vs Social Score</a></li>
        <li><a href="#ah" style="text-decoration:none; color:#007bff;">ESG Score vs Governance Score</a></li>
    </ul>
</div>
""", unsafe_allow_html=True)


# Function to create local filters for each chart
def apply_local_filter(dataframe, filter_label):
    local_company_filter = st.multiselect(f"Search and Add Companies for {filter_label}", options=dataframe["Company"].unique(), default=[])
    local_type_filter = st.multiselect(f"Search and Add Types for {filter_label}", options=dataframe["Type"].unique(), default=[])

    if local_company_filter:
        dataframe = dataframe[dataframe["Company"].isin(local_company_filter)]
    if local_type_filter:
        dataframe = dataframe[dataframe["Type"].isin(local_type_filter)]

    return dataframe


st.markdown("### Environmental Score Comparison")
df_env_filtered = apply_local_filter(df, "Environmental Score Comparison")
if df_env_filtered.empty:
    st.warning("No data available for Environmental Score Comparison.")
else:
    fig_env = px.bar(
        df_env_filtered,
        x="Company",
        y="Environmental Score",
        color="Type",
        title="Environmental Score by Company",
        labels={"Environmental Score": "Environmental Score", "Company": "Company Name"},
        hover_data=["Social Score", "Governance Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_env.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_env, use_container_width=False)


st.markdown("### Social Score Comparison")
df_soc_filtered = apply_local_filter(df, "Social Score Comparison")
if df_soc_filtered.empty:
    st.warning("No data available for Social Score Comparison.")
else:
    fig_soc = px.bar(
        df_soc_filtered,
        x="Company",
        y="Social Score",
        color="Type",
        title="Social Score by Company",
        labels={"Social Score": "Social Score", "Company": "Company Name"},
        hover_data=["Environmental Score", "Governance Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_soc.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_soc, use_container_width=False)


st.markdown("### Governance Score Comparison")
df_gov_filtered = apply_local_filter(df, "Governance Score Comparison")
if df_gov_filtered.empty:
    st.warning("No data available for Governance Score Comparison.")
else:
    fig_gov = px.bar(
        df_gov_filtered,
        x="Company",
        y="Governance Score",
        color="Type",
        title="Governance Score by Company",
        labels={"Governance Score": "Governance Score", "Company": "Company Name"},
        hover_data=["Environmental Score", "Social Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_gov.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_gov, use_container_width=False)

# 3D Scatter Plot for Environmental, Social, and Governance Scores
st.markdown("### 3D ESG Scatter Plot")
df_3d_filtered = apply_local_filter(df, "3D ESG Scatter Plot")
if df_3d_filtered.empty:
    st.warning("No data available for 3D ESG Scatter Plot.")
else:
    fig_3d = px.scatter_3d(
        df_3d_filtered,
        x="Environmental Score",
        y="Social Score",
        z="Governance Score",
        color="Type",
        title="3D Scatter Plot of ESG Scores",
        labels={"Environmental Score": "Environmental", "Social Score": "Social", "Governance Score": "Governance"},
        hover_data=["Company"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_3d.update_layout(height=700, width=1000,
        scene=dict(
        xaxis=dict(title="Environmental Score"),
        yaxis=dict(title="Social Score"),
        zaxis=dict(title="Governance Score")
    ))
    st.plotly_chart(fig_3d, use_container_width=False)

st.markdown('<a id="radar-chart"></a>', unsafe_allow_html=True)
st.markdown("### Radar Chart for ESG Scores")
df_radar_filtered = apply_local_filter(df, "Radar Chart")
if df_radar_filtered.empty:
    st.warning("No data available for Radar Chart.")
else:
    radar_fig = go.Figure()
    for _, row in df_radar_filtered.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=[row["Environmental Score"], row["Social Score"], row["Governance Score"], row["Environmental Score"]],
            theta=["Environmental", "Social", "Governance", "Environmental"],
            fill='toself',
            name=row["Company"]
        ))
    radar_fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        title="Radar Chart of ESG Scores by Company",
        height=700, 
        width=1000,
        template="plotly_dark",
        showlegend=False,
    )
    st.plotly_chart(radar_fig, use_container_width=False)



# Environmental vs Social Scores
st.markdown('<a id="aa"></a>', unsafe_allow_html=True)
st.markdown("### Enhanced Environmental vs Social Scores")
df_es_filtered = apply_local_filter(df, "Environmental vs Social Scores")
if df_es_filtered.empty:
    st.warning("No data available for Environmental vs Social Scores.")
else:
    fig_es = px.scatter(
        df_es_filtered,
        x="Environmental Score",
        y="Social Score",
        color="Type",
        title="Environmental vs Social Scores",
        hover_data=["Company", "Governance Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_es.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGray")))
    fig_es.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_es, use_container_width=False)

# Enhanced Environmental vs Governance Scores
st.markdown('<a id="ab"></a>', unsafe_allow_html=True)
st.markdown("### Enhanced Environmental vs Governance Scores")
df_eg_filtered = apply_local_filter(df, "Environmental vs Governance Scores")
if df_eg_filtered.empty:
    st.warning("No data available for Environmental vs Governance Scores.")
else:
    fig_eg = px.scatter(
        df_eg_filtered,
        x="Environmental Score",
        y="Governance Score",
        color="Type",
        title="Environmental vs Governance Scores",
        hover_data=["Company", "Social Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_eg.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGray")))
    fig_eg.update_layout(height=700, width=1000, template="plotly_white")
    st.plotly_chart(fig_eg, use_container_width=False)

# Social vs Governance Scores
st.markdown('<a id="ac"></a>', unsafe_allow_html=True)
st.markdown("### Enhanced Social vs Governance Scores")
df_sg_filtered = apply_local_filter(df, "Social vs Governance Scores")
if df_sg_filtered.empty:
    st.warning("No data available for Social vs Governance Scores.")
else:
    fig_sg = px.scatter(
        df_sg_filtered,
        x="Social Score",
        y="Governance Score",
        color="Type",
        title="Social vs Governance Scores",
        hover_data=["Company", "Environmental Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_sg.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGray")))
    fig_sg.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_sg, use_container_width=False)

# Stacked Bar Chart of ESG Scores
st.markdown('<a id="ad"></a>', unsafe_allow_html=True)
st.markdown("### Stacked Bar Chart of ESG Scores by Company")
df_stacked_filtered = apply_local_filter(df, "Stacked Bar Chart")
if df_stacked_filtered.empty:
    st.warning("No data available for Stacked Bar Chart.")
else:
    stacked_bar = px.bar(
        df_stacked_filtered.melt(id_vars=["Company"], value_vars=["Environmental Score", "Social Score", "Governance Score"],
                var_name="Score Type", value_name="Score"),
        x="Company",
        y="Score",
        color="Score Type",
        title="Stacked Bar Chart of ESG Scores by Company",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    stacked_bar.update_layout(height=800, width=1200, template="plotly_dark")
    st.plotly_chart(stacked_bar, use_container_width=False)

# Histogram of ESG Scores
st.markdown('<a id="ae"></a>', unsafe_allow_html=True)
st.markdown("### Histogram of ESG Scores")
df_histogram_filtered = apply_local_filter(df, "Histogram of ESG Scores")
if df_histogram_filtered.empty:
    st.warning("No data available for Histogram.")
else:
    histogram = px.histogram(
        df_histogram_filtered.melt(id_vars=["Type"], value_vars=["Environmental Score", "Social Score", "Governance Score"],
                var_name="Score Type", value_name="Score"),
        x="Score",
        color="Score Type",
        barmode="overlay",
        title="Distribution of ESG Scores",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    histogram.update_layout(height=700, width=1000, template="plotly_white")
    st.plotly_chart(histogram, use_container_width=False)



st.markdown('<a id="af"></a>', unsafe_allow_html=True)
st.markdown("### ESG Score vs Environmental Score")
df_esg_env_filtered = apply_local_filter(df, "ESG Score vs Environmental Score")
if df_esg_env_filtered.empty:
    st.warning("No data available for ESG Score vs Environmental Score.")
else:
    fig_esg_env = px.scatter(
        df_esg_env_filtered,
        x="Environmental Score",
        y="ESG Score",
        color="Type",
        title="ESG Score vs Environmental Score",
        labels={"Environmental Score": "Environmental Score", "ESG Score": "ESG Score"},
        hover_data=["Company", "Social Score", "Governance Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_esg_env.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_esg_env, use_container_width=False)

st.markdown('<a id="ag"></a>', unsafe_allow_html=True)
st.markdown("### ESG Score vs Social Score")
df_esg_social_filtered = apply_local_filter(df, "ESG Score vs Social Score")
if df_esg_social_filtered.empty:
    st.warning("No data available for ESG Score vs Social Score.")
else:
    fig_esg_social = px.scatter(
        df_esg_social_filtered,
        x="Social Score",
        y="ESG Score",
        color="Type",
        title="ESG Score vs Social Score",
        labels={"Social Score": "Social Score", "ESG Score": "ESG Score"},
        hover_data=["Company", "Environmental Score", "Governance Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_esg_social.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_esg_social, use_container_width=False)


st.markdown('<a id="ah"></a>', unsafe_allow_html=True)
st.markdown("### ESG Score vs Governance Score")
df_esg_gov_filtered = apply_local_filter(df, "ESG Score vs Governance Score")
if df_esg_gov_filtered.empty:
    st.warning("No data available for ESG Score vs Governance Score.")
else:
    fig_esg_gov = px.scatter(
        df_esg_gov_filtered,
        x="Governance Score",
        y="ESG Score",
        color="Type",
        title="ESG Score vs Governance Score",
        labels={"Governance Score": "Governance Score", "ESG Score": "ESG Score"},
        hover_data=["Company", "Environmental Score", "Social Score"],
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_esg_gov.update_layout(height=700, width=1000, template="plotly_dark")
    st.plotly_chart(fig_esg_gov, use_container_width=False)





