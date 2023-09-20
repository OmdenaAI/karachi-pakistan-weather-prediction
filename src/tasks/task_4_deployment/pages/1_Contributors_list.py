import streamlit as st

st.set_page_config(
    page_title='Weather Prediction',
    page_icon=':sun_behind_cloud',
    initial_sidebar_state='auto',
)

st.title('Karachi Pakistan Weather Prediction', anchor=None)
st.subheader('Active Contributors List')

st.markdown(
"""
- Aniket Bhausaheb Barphe
- Marc Kolb
- Ye Bhone Lin
- Soh Zong Xian
  
"""
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)