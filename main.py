from agent import agent_with_chat_history
import streamlit as st

input = st.text_area(label="Input", key="input", height=500, placeholder="Input what you need to rewrite")
btn = st.button(label="Generate")

if btn and input != "":
    res = agent_with_chat_history.invoke(
        {"name": "John", "input": input},
        config={"configurable": {"session_id": "<foo>"}})
    st.write(res['output'])

