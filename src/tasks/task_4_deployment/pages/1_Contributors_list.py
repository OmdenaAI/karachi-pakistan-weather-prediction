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
- Valéry Bonneau
- Utkarsh Trivedi
- Nick Schilders
- Kamal Muhamed
- Sidra Tul Muntaha 
- Sai Nagender Sama
- Piyush rawat 
- Muhammad Azam
- Augustine M Gbondo
- Talha Mehfooz Khursaidi 
- Rohit Singh 
- Ogbonnaya Ogah
- Sunil Zakane 
- Sahar Nikoo
- Kelvin Njenga
- N R Sailesh
  
"""
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)