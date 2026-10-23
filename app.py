import streamlit as st

st.set_page_config(
    page_title="StockLens",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("pages/1_Overview.py", title="Overview", default=True),
    st.Page("pages/2_Stock_Analysis.py", title="Stock Analysis"),
    st.Page("pages/3_ML_Prediction.py", title="ML Prediction"),
    st.Page("pages/4_Fundamentals.py", title="Fundamentals"),
    st.Page("pages/5_News_Sentiment.py", title="News & Sentiment"),
    st.Page("pages/6_AI_Insights.py", title="AI Insights"),
    st.Page("pages/7_Compare_Stocks.py", title="Compare Stocks"),
    st.Page("pages/8_Methodology.py", title="Methodology"),
]

navigation = st.navigation(pages)
navigation.run()
