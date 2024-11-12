import streamlit as st

#Sample
# def page_2():
#     st.title("Page2")
# pg = st.navigation([st.Page("Introduction.py"), st.Page(page_2)])

# #Navigation
pg = st.navigation([st.Page("Introduction.py"), st.Page("Models.py"), st.Page("Charts.py")])
pg.run()


