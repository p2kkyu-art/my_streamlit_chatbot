"""간단한 Streamlit 웹 챗봇.

API 키는 C:\\env\\.env 의 OPENAI_API_KEY 를 읽고,
대화는 OpenAI Responses API(gpt-4o-mini)로 전달합니다.
"""

import os
from pathlib import Path
from typing import Literal, TypedDict

import streamlit as st
from openai import OpenAI, OpenAIError

try:
    from dotenv import load_dotenv
except ImportError:  # Render에 패키지가 없어도 환경 변수만으로 동작합니다.
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

ENV_PATH = Path(r"C:\env\.env")
MODEL = "gpt-4o-mini"
INSTRUCTIONS = "You are a helpful assistant. Reply in the same language as the user."


class ChatMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


def load_api_key() -> str | None:
    """C:\\env\\.env 에서 OPENAI_API_KEY 를 읽습니다."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    return api_key or None


@st.cache_resource
def get_openai_client() -> OpenAI:
    """프로세스 안에서 OpenAI 클라이언트를 한 번만 만듭니다."""
    api_key = os.environ["OPENAI_API_KEY"]
    return OpenAI(api_key=api_key)


def stream_reply(client: OpenAI, messages: list[ChatMessage]) -> str:
    """대화 기록을 모델에 보내고, 스트리밍된 텍스트를 그대로 반환합니다."""
    stream = client.responses.create(
        model=MODEL,
        input=list(messages),
        instructions=INSTRUCTIONS,
        store=False,
        stream=True,
    )
    reply = st.write_stream(stream)
    if isinstance(reply, str):
        return reply
    return "".join(part for part in reply if isinstance(part, str))


st.set_page_config(page_title="웹 챗봇", page_icon=":material/chat:")
st.title(":material/chat: 웹 챗봇")
st.caption(f"모델 {MODEL}")

api_key = load_api_key()
if api_key is None:
    st.error(
        "OPENAI_API_KEY가 없습니다. 로컬은 C:\\env\\.env, Render는 Environment에 넣어 주세요.",
        title="API 키 없음",
    )
    st.stop()

st.session_state.setdefault("messages", [])
messages: list[ChatMessage] = st.session_state.messages

raw_prompt = st.chat_input("메시지를 입력하세요", submit_mode="disable")
prompt = raw_prompt.strip() if isinstance(raw_prompt, str) else ""

if not messages and not prompt:
    st.info("아래 입력창에 메시지를 보내고 전송 버튼을 누르세요.", title="대화 시작")

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not prompt:
    st.stop()

user_message: ChatMessage = {"role": "user", "content": prompt}
messages.append(user_message)
with st.chat_message("user"):
    st.markdown(user_message["content"])

with st.chat_message("assistant"):
    try:
        answer = stream_reply(get_openai_client(), messages)
    except OpenAIError as error:
        st.error(str(error), title="응답 실패")
        st.stop()

if not answer.strip():
    st.warning("모델이 빈 응답을 반환했습니다.", title="빈 응답")
    st.stop()

messages.append({"role": "assistant", "content": answer})
