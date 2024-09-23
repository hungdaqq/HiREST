import streamlit as st

# Display custom CSS to change the cursor
st.markdown(
    """
    <style>
    body {
        cursor: url('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQwLciabHzh63qsdR_8Wz-TSil9X19ZtgVUaA&s'), auto; /* Replace with your cursor URL */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your Streamlit app content
st.title("Custom Mouse Avatar in Streamlit")
st.write("Hover over this page to see the custom mouse avatar!")
