from agent import agent
import streamlit as st

input_url = st.text_input(label="Input", key="input", placeholder="Input what you need to rewrite")

col1, col2, col3 = st.columns(3)

with col1:
    title_character_count = st.text_input(label="title character count", key="title_character_count",
                                          placeholder="title character count")

with col2:
    desc_character_count = st.text_input(label="desc character count", key="desc_character_count",
                                         placeholder="desc character count")

with col3:
    body_word_count = st.text_input(label="body word count", key="body_word_count",
                                    placeholder="body word count")

btn = st.button(label="Generate")

if btn and input_url != "":
    res = agent(urls=input_url,
                title_character_count=title_character_count or 60,
                desc_character_count=desc_character_count or 160,
                body_word_count=body_word_count or 500)
    st.write(res['output'])
