import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="教えて！えどがわ区議会AI",
    page_icon="🦉",
    layout="wide",
)

components.html(
    """
    <script>
    window.parent.document.documentElement.lang = 'ja';
    </script>
    """,
    height=0,
)

MD_FILE_PATH = os.path.join(
    os.path.dirname(__file__),
    "output",
    "cleaned_全審査.md",
)

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("APIキーが設定されていません。`.streamlit/secrets.toml` に `GEMINI_API_KEY` を設定してください。")
    st.stop()


@st.cache_resource(show_spinner="議事録を読み込み中...")
def build_vectorstore(api_key: str):
    loader = TextLoader(MD_FILE_PATH, encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
    )
    return FAISS.from_documents(chunks, embeddings)


vectorstore = build_vectorstore(api_key)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは江戸川区議会 予算特別委員会（令和8年度）の議事録に基づいて回答するアシスタントです。
以下の議事録の抜粋を参考にして、質問に日本語で正確に答えてください。
抜粋に記載のない内容については「議事録には記載がありません」と答えてください。
議事録には発言者名（例：「〇〇委員」「△△課長」）が記載されています。回答する際は、誰がどのような発言をしたのか、発言者名を積極的に引用してください。

【議事録の抜粋】
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔰 教えて！えどがわ議会AI 🦉")
st.caption("「区議会って難しそう…」を解決！令和8年度の予算審査でどんな話し合いがあったのか、AIがやさしくお答えします✨　作成者：[あき@データで見る江戸川区](https://x.com/edogawa_aki)")

AVATARS = {"user": "👤", "assistant": "🦉"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

if not st.session_state.messages:
    st.markdown("💡 **まずは、気になるボタンをタップしてみてね！**")
    col1, col2, col3 = st.columns(3)
    suggest_question = None
    if col1.button("👶 子育て支援について"):
        suggest_question = "子育て支援についてどのような議論がありましたか？"
    if col2.button("🏫 小中学校の環境について"):
        suggest_question = "小中学校の環境についてどのような議論がありましたか？"
    if col3.button("🔥 どんな話題が白熱している？"):
        suggest_question = "どんな話題について特に活発な議論や白熱したやり取りがありましたか？発言者名も教えてください。"
    if suggest_question:
        st.session_state._suggest = suggest_question
        st.rerun()

chat_input_question = st.chat_input(
    "質問してみてね（例：予算審査で議論された主な項目は？　〇〇議員はどんな質問をしている？）"
)
if "_suggest" in st.session_state:
    question = st.session_state.pop("_suggest")
elif chat_input_question:
    question = chat_input_question
else:
    question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    chat_history = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history.append(HumanMessage(content=m["content"]))
        else:
            chat_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant", avatar="🦉"):
        with st.spinner("回答を生成中..."):
            try:
                docs = retriever.invoke(question)
                context = format_docs(docs)
                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "chat_history": chat_history,
                    "question": question,
                })
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("申し訳ありません。回答の生成中にエラーが発生しました。時間をおいて再度お試しください。")
                print(f"[ERROR] {e}")
