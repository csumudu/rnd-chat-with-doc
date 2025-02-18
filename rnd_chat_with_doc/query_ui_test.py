import streamlit as st
from llama_index.core.chat_engine.types import ChatMode
from rnd_chat_with_doc.modules.post_proessors.score_filter_processor import (
    ScoreNodePostprocessor,
)
from rnd_chat_with_doc.modules.util.llm_manager import get_nim_llm
from rnd_chat_with_doc.query_test import get_index
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.postprocessor import SentenceEmbeddingOptimizer


@st.cache_resource(show_spinner=False)
def get_index_wrapper():
    llm = OpenAI(
        temperature=0,
        api_base="http://localhost:5000/v1",
        api_key="test",
        timeout=900,
        max_retries=0,
    )

    Settings.llm = llm
    return get_index()


st.set_page_config(
    page_title="CC Assistant",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items=None,
    page_icon="🍾",
)

index, db = get_index_wrapper()
if "chatEngine" not in st.session_state.keys():
    st.session_state.chatEngine = index.as_chat_engine(
        chat_mode=ChatMode.CONTEXT,
        verbose=True,
        node_postprocessors=[
            SentenceEmbeddingOptimizer(percentile_cutoff=0.7),
            ScoreNodePostprocessor(),
        ],
    )

st.title("Customer Connect CSR Assistant")

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
        # Non Stream response
        # with st.spinner("Please wait......"):
        #     res = st.session_state.chatEngine.chat(message=prompt)
        #     st.write(res.response)
        #     msg = {"role": "assistant", "content": res.response}
        #     st.session_state.messages.append(msg)
        response = st.session_state.chatEngine.stream_chat(message=prompt)
        nodes = [node for node in response.source_nodes]
        for (
            col,
            node,
        ) in zip(st.columns(len(nodes)), nodes):
            with col:
                st.header(f"score->{node.score}")
                st.write(node.text)

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
