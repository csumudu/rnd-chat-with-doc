import streamlit as st
from rnd_chat_with_doc.modules.tools.cc_doc_tool import (
    character_counter,
    customer_connect_documentation,
)
from rnd_chat_with_doc.modules.util.llama_index_utils import set_llamaindex_logging
from rnd_chat_with_doc.query_test import get_index
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.chat_engine.types import ChatMode
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool


@st.cache_resource(show_spinner=False)
def get_agent():
    set_llamaindex_logging()

    llm = OpenAI(
        temperature=0,
        api_base="http://localhost:5000/v1",
        api_key="test",
        timeout=900,
        max_retries=0,
    )

    Settings.llm = llm
    t1 = FunctionTool.from_defaults(
        fn=customer_connect_documentation, name="Documentation"
    )
    t2 = FunctionTool.from_defaults(
        fn=character_counter, name="Count Characters"
    )

    agent = OpenAIAgent.from_tools(tools=[t1, t2], llm=llm, verbose=True)
    return agent


if "agent" not in st.session_state.keys():
    st.session_state.agent = get_agent()

st.set_page_config(
    page_title="CC Agent",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
    page_icon="😂",
)

st.title("Customer Connect Agent")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi!.. How can I help you today?"}
    ]

if prompt := st.chat_input("Question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg.get("content"))


if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = st.session_state.agent.stream_chat(message=prompt)
        message_placeholder = st.empty()
        full_response = ""

        with st.spinner("Please wait......"):
            first_chunk = next(response.response_gen)
            full_response += first_chunk
            message_placeholder.markdown(full_response + "▌")

        for chunk in response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        msg = {"role": "assistant", "content": full_response}
        message_placeholder.markdown(full_response)
        st.session_state.messages.append(msg)
