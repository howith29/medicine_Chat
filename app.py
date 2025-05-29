import streamlit as st

st.title("Hello, Streamlit!")
st.write("이건 간단한 텍스트입니다.")

name = st.text_input("이름 입력하세요:")
if name:
    st.write(f"{name}님, 안녕하세요")