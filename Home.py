import streamlit as st


# Set up page configuration
st.set_page_config(
    page_title="Ecologic",  # New app name
    page_icon="🌱",
    layout="wide",
)

# Title and Welcome Message
st.title("Ecologic ")
st.markdown(
    """
    ## A Smarter Way to Measure Sustainability 🌱
    """
)

# Add a banner image (optional)
st.image(
    "https://web.cdn.crystalfunds.com/public-web/the-ones/20190301/images/2022/08-august/esg/esg--og.png", 
    caption="Sustainability in Action", 
    use_container_width=True
)

# Sidebar Navigation Info
st.markdown(
    """
    ### 🌟 Features
    Navigate through the app using the **sidebar**:
    - **📊 ESG Calculator**: Input industrial data to compute your ESG score and monitor your impact.
    - **📈 Visualization**: Visualize detailed ESG metrics and current scenarios across industries.
    - **📈 Press Releases**: Explore detailed trends, comparisons, and NEWS into ESG metrics across industries.
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
    """
)
