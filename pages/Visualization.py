import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.title("Enhanced ESG Ratings Visualization")

# File path to the CSV
csv_file_path = "merged.csv"

# Load the CSV file
@st.cache_data
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
}
df.rename(columns=column_mapping, inplace=True)


# Ensure 'Type' exists
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

# Add new company if not found
if company_name and (matching_companies.empty or "selected_company" not in st.session_state):
    company_type = st.selectbox("Select Company Type", options=types)

    st.markdown("### Enter ESG Scores")
    environmental_score = st.slider("Environmental Score (0-100)", 0, 100, 50)
    social_score = st.slider("Social Score (0-100)", 0, 100, 50)
    governance_score = st.slider("Governance Score (0-100)", 0, 100, 50)

    if st.button("Save New Company"):
        overall_score = (environmental_score + social_score + governance_score) / 3

        # Calculate ESG Rating
        def get_esg_rating(score):
            if score >= 90:
                return "AAA"
            elif score >= 80:
                return "AA"
            elif score >= 70:
                return "A"
            elif score >= 60:
                return "BBB"
            elif score >= 50:
                return "BB"
            elif score >= 40:
                return "B"
            elif score >= 30:
                return "CCC"
            else:
                return "D"

        rating = get_esg_rating(overall_score)

        new_data = {
            "Company": company_name,
            "Type": company_type,
            "Environmental Score": environmental_score,
            "Social Score": social_score,
            "Governance Score": governance_score,
            "Overall Score": overall_score,
            "Rating": rating,
        }

        # Append to the CSV file
        new_df = pd.DataFrame([new_data])
        new_df.to_csv(csv_file_path, mode="a", header=False, index=False)

        st.success(f"New company '{company_name}' added successfully!")
        # Reload the dataframe after update
        df = load_csv(csv_file_path)

# Display Saved Results
st.markdown("### Saved Results")
st.dataframe(df)

# Add navigation links with improved styling
st.markdown("### Graph Navigation")
st.markdown("""
<div style="background-color:#f0f0f0; padding:10px; border-radius:5px;">
    <ul style="list-style-type:none; padding-left:0;">
        <li><a href="#environmental-score-comparison" style="text-decoration:none; color:#007bff;">🌿 Environmental Score Comparison</a></li>
        <li><a href="#social-score-comparison" style="text-decoration:none; color:#007bff;">🤝 Social Score Comparison</a></li>
        <li><a href="#governance-score-comparison" style="text-decoration:none; color:#007bff;">🏛️ Governance Score Comparison</a></li>
        <li><a href="#3d-esg-scatter-plot" style="text-decoration:none; color:#007bff;">📊 3D ESG Scatter Plot</a></li>
        <li><a href="#radar-chart" style="text-decoration:none; color:#007bff;">📈 Radar Chart</a></li>
        <li><a href="#heatmap" style="text-decoration:none; color:#007bff;">🔥 Heatmap</a></li>
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

# Enhanced Environmental Score Comparison
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
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_env.update_layout(height=700, width=1200, template="plotly_dark")
    st.plotly_chart(fig_env, use_container_width=False)

# Enhanced Social Score Comparison
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
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_soc.update_layout(height=700, width=1200, template="plotly_white")
    st.plotly_chart(fig_soc, use_container_width=False)

# Enhanced Governance Score Comparison
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
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_gov.update_layout(height=700, width=1200, template="plotly_dark")
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
    fig_3d.update_layout(height=800, width=1200, scene=dict(
        xaxis=dict(title="Environmental Score"),
        yaxis=dict(title="Social Score"),
        zaxis=dict(title="Governance Score")
    ))
    st.plotly_chart(fig_3d, use_container_width=False)

# Enhanced Radar Chart for ESG Scores
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Radar Chart of ESG Scores by Company",
        height=800,
        width=1200,
        template="plotly_dark",
        showlegend=True
    )
    st.plotly_chart(radar_fig, use_container_width=False)

# Distribution of Scores by Type
st.markdown("### Distribution of ESG Scores by Company Type")
df_distribution_filtered = apply_local_filter(df, "Distribution of ESG Scores")
if df_distribution_filtered.empty:
    st.warning("No data available for Distribution of ESG Scores.")
else:
    fig_box = px.box(
        df_distribution_filtered.melt(id_vars=["Company", "Type"], value_vars=["Environmental Score", "Social Score", "Governance Score"],
                var_name="Score Type", value_name="Score"),
        x="Score Type",
        y="Score",
        color="Type",
        title="Distribution of ESG Scores by Company Type",
        labels={"Score": "Score", "Score Type": "Type of Score"},
        points="all",  # Show individual points
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig_box.update_layout(height=700, width=1200, template="plotly_white")
    st.plotly_chart(fig_box, use_container_width=False)

# Environmental vs Social Scores
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
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig_es.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGray")))
    fig_es.update_layout(height=800, width=1200, template="plotly_white")
    st.plotly_chart(fig_es, use_container_width=False)

# Environmental vs Governance Scores
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
        color_discrete_sequence=px.colors.sequential.Magma
    )
    fig_eg.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGray")))
    fig_eg.update_layout(height=800, width=1200, template="plotly_white")
    st.plotly_chart(fig_eg, use_container_width=False)

# Social vs Governance Scores
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
        color_discrete_sequence=px.colors.sequential.Cividis
    )
    fig_sg.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="DarkSlateGray")))
    fig_sg.update_layout(height=800, width=1200, template="plotly_white")
    st.plotly_chart(fig_sg, use_container_width=False)

# Heatmap of ESG Scores
st.markdown("### Heatmap of ESG Scores")
df_heatmap_filtered = apply_local_filter(df, "Heatmap")
if df_heatmap_filtered.empty:
    st.warning("No data available for Heatmap.")
else:
    heatmap_data = df_heatmap_filtered.melt(id_vars=["Company"], value_vars=["Environmental Score", "Social Score", "Governance Score"],
                           var_name="Score Type", value_name="Score")
    fig_heatmap = px.density_heatmap(
        heatmap_data,
        x="Score Type",
        y="Company",
        z="Score",
        color_continuous_scale=px.colors.sequential.Plasma,
        title="Heatmap of ESG Scores by Company"
    )
    fig_heatmap.update_layout(height=800, width=1200, template="plotly_white")
    st.plotly_chart(fig_heatmap, use_container_width=False)

# Pie Chart of Company Types
st.markdown("### Distribution of Companies by Type")
df_pie_filtered = apply_local_filter(df, "Pie Chart of Companies by Type")
if df_pie_filtered.empty:
    st.warning("No data available for Pie Chart.")
else:
    pie_chart = px.pie(
        df_pie_filtered,
        names="Type",
        title="Distribution of Companies by Type",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    pie_chart.update_layout(height=700, width=800, template="plotly_white")
    st.plotly_chart(pie_chart, use_container_width=False)

# Stacked Bar Chart of ESG Scores
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
        color_discrete_sequence=px.colors.sequential.Rainbow
    )
    stacked_bar.update_layout(height=800, width=1200, template="plotly_white")
    st.plotly_chart(stacked_bar, use_container_width=False)

# Histogram of ESG Scores
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
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    histogram.update_layout(height=700, width=1200, template="plotly_white")
    st.plotly_chart(histogram, use_container_width=False)
