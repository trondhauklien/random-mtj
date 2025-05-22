import streamlit as st


pages = {
    "Examples": [
        st.Page("sandbox.py", title="Convergence"),
        st.Page("precession.py", title="Precession"),
        st.Page("stt-switching.py", title="STT Switching"),
    ],
}

pg = st.navigation(pages)
pg.run()
