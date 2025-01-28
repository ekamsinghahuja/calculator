import streamlit as st

# Set up page configuration
st.set_page_config(
    page_title="SimplicatION",  # New app name
    page_icon="🌱",
    layout="wide",
)

# Title and Welcome Message
st.title("SimplicatION (simple solutions for sustainability)")
st.markdown(
    """
    ## A Smarter Way to Measure Sustainability 🌱
    **EcoImpact Analyzer** is your one-stop platform for calculating and visualizing ESG (Environmental, Social, and Governance) metrics. Whether you're a business owner, an analyst, or just an enthusiast, explore how industries are driving sustainable practices.
    """
)

# Add a banner image (optional)
st.image(
    "https://images.unsplash.com/photo-1542831371-d531d36971e6", 
    caption="Sustainability in Action", 
    use_column_width=True
)

# Sidebar Navigation Info
st.markdown(
    """
    ### 🌟 Features
    Navigate through the app using the **sidebar**:
    - **📊 ESG Calculator**: Input industrial data to compute your ESG score and monitor your impact.
    - **📈 Visualization**: Explore detailed trends, comparisons, and insights into ESG metrics across industries.
    """
)

# Add a call-to-action button
if st.button("Get Started Now 🚀"):
    st.write("👉 Use the sidebar to begin your journey!")

# Add a footer with credits
st.markdown(
    """
    ---
    💡 **About This App**: Created to inspire sustainable practices through data-driven insights.  
    🌟 **Credits**: Powered by Python & Streamlit | Developed by [Your Name].  
    """
)
