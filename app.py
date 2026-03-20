import os
import gspread
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
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

【回答のルール】
- 議事録に記載されている発言者の実際の名前（議員名・職員名）をそのまま使用してください。名前を伏せたり、プレースホルダーに置き換えたりしないでください。
- 議事録の言葉をできるだけそのまま引用し、「〜と述べました」「〜を意見しました」のような抽象的なまとめは避けてください。
- 「誰が何を質問し、誰がどう答えたか」という質疑応答の流れが伝わるように記述してください。
- 予算額・事業名・施策名など具体的な数値や固有名詞があれば必ず含めてください。
- 複数の論点がある場合は箇条書きで整理してください。

【議事録の抜粋】
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def save_log(question, answer):
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
        worksheet = sh.get_worksheet(0)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([now, question, answer])
    except Exception as e:
        print("Logging error: failed to write to spreadsheet")


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔰 教えて！えどがわ議会AI 🦉")
st.caption("「区議会って難しそう…」を解決！令和8年度の予算審査でどんな話し合いがあったのか、AIがやさしくお答えします✨　作成者：[あき@データで見る江戸川区](https://x.com/edogawa_aki)")

AVATARS = {"user": "👤", "assistant": "🦉"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

if not st.session_state.messages and "_suggest" not in st.session_state:
    st.markdown("💡 **まずは、気になるボタンをタップしてみてね！**")
    suggest_question = None
    if st.button("👶 子育て支援について"):
        suggest_question = "子育て支援についてどのような議論がありましたか？"
    if st.button("🏫 小中学校の環境について"):
        suggest_question = "小中学校の環境についてどのような議論がありましたか？"
    if st.button("🔥 どんな話題が白熱している？"):
        suggest_question = "どんな話題について特に活発な議論や白熱したやり取りがありましたか？発言者名も教えてください。"
    if st.button("💻 デジタル化・DXの進展について"):
        suggest_question = "デジタル化・DXの進展についてはどのような議論がありましたか？"
    if st.button("👴 高齢者福祉について"):
        suggest_question = "高齢者福祉についてはどのような議論がありましたか？"
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
                save_log(question, answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error("申し訳ありません。回答の生成中にエラーが発生しました。時間をおいて再度お試しください。")
                print("[ERROR] Answer generation failed")
