import streamlit as st


pages = {
    "Examples": [
        st.Page("sandbox.py", title="Sandbox"),
        st.Page("precession.py", title="Precession"),
    ],
}

pg = st.navigation(pages)
pg.run()
