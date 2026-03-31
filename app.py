import os
import re
import uuid
import gspread
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
from streamlit_cookies_manager import CookieManager
from google.oauth2 import service_account
from google.cloud import firestore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_google_firestore import FirestoreVectorStore
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

st.markdown("""
<style>
[data-testid="stChatMessage"] h1 { font-size: 1.4rem; }
[data-testid="stChatMessage"] h2 { font-size: 1.2rem; }
[data-testid="stChatMessage"] h3 { font-size: 1.05rem; }
[data-testid="stChatMessage"] h4 { font-size: 1.0rem; }
</style>
""", unsafe_allow_html=True)

try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("APIキーが設定されていません。`.streamlit/secrets.toml` に `GEMINI_API_KEY` を設定してください。")
    st.stop()


class FixedDimEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts, **kwargs):
        return super().embed_documents(texts, output_dimensionality=768, **kwargs)
    def embed_query(self, text, **kwargs):
        return super().embed_query(text, output_dimensionality=768, **kwargs)


@st.cache_resource(show_spinner=False)
def get_retriever(api_key: str):
    embeddings = FixedDimEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        task_type="RETRIEVAL_QUERY",
    )
    creds = service_account.Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"])
    )
    db = firestore.Client(credentials=creds, project=creds.project_id)
    with st.spinner("📂 インデックスを読み込み中..."):
        vectorstore = FirestoreVectorStore(
            collection="edogawa_gijiroku",
            embedding_service=embeddings,
            client=db,
        )
    return vectorstore.as_retriever(search_kwargs={"k": 10})


cookies = CookieManager()
if not cookies.ready():
    st.stop()
if "user_id" not in cookies:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

retriever = get_retriever(api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは江戸川区議会の議事録を読み解き、区民に分かりやすく立体的に解説する、優秀なWebメディア編集長（えどがわ議会AI）です。
知識ベースには、令和8年度予算特別委員会（最新）と、令和5年度以降の各種会議の議事録が含まれています。
以下の議事録の抜粋を参考にして、質問に日本語で正確に答えてください。記載のない内容は「議事録には記載がありません」と答えてください。

【回答の構成とルール（厳守）】
冒頭の「承知いたしました」といった挨拶や自己紹介は一切不要です。質問の意図を汲み取り、以下の1〜4の構成で出力してください。

1. 🔥 30秒でわかる！今回のハイライト
質問に対する結論や、一番の「見どころ」を3〜4行でキャッチーに要約してください。

2. 🔍 注目のポイント（※質問のタイプに合わせて柔軟に構成を変更すること！）
ユーザーの質問タイプに合わせて、以下のいずれか最も適したアプローチで解説を展開してください。見出しもWeb記事風に魅力的なものにします。

- パターンA【特定のテーマ・施策についての質問の場合】
  見出し例：「⚖️ 賛成 vs 反対！議会での主な意見」
  推進・賛成する意見と、慎重・反対・懸念する意見（または現状評価と改善要求）を対比させて箇条書きで整理してください。
- パターンB【特定の人物（議員など）についての質問の場合】
  見出し例：「💡 〇〇委員のココに注目！独自の視点と切り込み」
  その人物が特に力を入れているテーマや、区の姿勢を鋭く問いただしたポイントを、テーマごとに整理して解説してください。
- パターンC【ざっくりとした質問（面白い議論ある？など）の場合】
  見出し例：「🏆 議会が白熱！注目のトピック（ピックアップ）」
  議事録の中で特に議論が白熱している、または区民の生活に直結する重要なトピックを2〜3個ピックアップし、それぞれ何が問題になっているかを解説してください。
  ⚠️【絶対遵守】「結論を出すか出さないか」「採決をどうするか」といった委員長等による『事務的な議事進行・手続き』に関するやり取りは、絶対にピックアップしないでください。必ず「税金の使い道」「福祉」「インフラ」など、区民生活に関わる具体的な政策テーマを選んでください。

3. 🗣️ 議会から飛び出した「生の声」
議事録の中から、特に印象的・感情的な「名言・パワーワード」をカギカッコ「」で抜き出し、誰の発言か（いつの会議か）を添えて紹介してください。

4. 📌 今後の注目ポイント
議論を踏まえ、江戸川区として今後どうしていく方針なのか、または何が未解決課題として残っているのかを2〜3行でまとめて締めくくってください。「いかがでしたか？」などの結びの言葉は不要です。

【見出し記法ルール（必須）】
- 1〜4の大セクション（🔥🔍🗣️📌）は必ず `##` で記述する（例：`## 🔥 30秒でわかる！今回のハイライト`）
- セクション2内の小見出し（パターンA/B/Cの例示見出し）は必ず `###` で記述する
- `#`（H1）は絶対に使用しない

【文章表現のルール】
- 最新の令和8年度の情報を最優先してください。
- 議員の名前（〇〇委員）と具体的な数字は必ず盛り込んでください。
- お役所言葉の単調な要約は避け、読者の目を引く魅力的な文章（「〜と鋭く指摘！」「果たして〜でしょうか？」など）にしてください。

【次の質問候補の生成（必須）】
回答の最後に、以下のルールに従って5つの質問候補を生成し、[NEXT_QUESTIONS] ブロックに出力してください。
- 候補1・2：今の回答内容をさらに深掘りする質問（具体的な数字や、別の会議での比較など）。
- 候補3：今の話題に対する「反対意見」や「慎重な意見」を尋ねる質問（議論の対立軸を見せるため）。
- 候補4・5：現在話題にしている分野とは「全く異なる分野」の、江戸川区議会で注目されているトピック（例：子育ての話なら防災やDXなど）。
形式：
[NEXT_QUESTIONS]
- 質問候補1
- 質問候補2
- 質問候補3
- 質問候補4
- 質問候補5
[/NEXT_QUESTIONS]

【議事録の抜粋】
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def save_log(question, answer, source="manual", user_id="", session_id=""):
    try:
        credentials = st.secrets["gcp_service_account"]
        gc = gspread.service_account_from_dict(credentials)
        sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
        worksheet = sh.get_worksheet(0)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([now, user_id, session_id, source, question, answer])
    except Exception as e:
        print("Logging error: failed to write to spreadsheet")


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔰 教えて！えどがわ議会AI 🦉")
st.markdown("「区議会って難しそう…」を解決！令和8年度の予算審査でどんな話し合いがあったのか、AIがやさしくお答えします✨")
st.markdown("作成者：[あき@データで見る江戸川区](https://x.com/edogawa_aki)")
st.info("🔒 **ログの収集について**\n\n入力内容はアプリ改善のため匿名で記録されます。個人情報は入力しないでください。", icon="ℹ️")

AVATARS = {"user": "👤", "assistant": "🦉"}

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
        st.markdown(message["content"])

if (st.session_state.get("next_questions")
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "assistant"):
    deep_qs = st.session_state.next_questions[:3]
    other_qs = st.session_state.next_questions[3:]
    st.markdown("**💡 関連する深掘り**")
    dq_cols = st.columns(len(deep_qs)) if len(deep_qs) > 1 else [st.container()]
    for i, nq in enumerate(deep_qs):
        with dq_cols[i]:
            if st.button(nq, key=f"next_q_hist_{i}", use_container_width=True):
                st.session_state._suggest = nq
                st.session_state._suggest_source = "suggest_next"
                del st.session_state["next_questions"]
                st.rerun()
    if other_qs:
        st.markdown("**🔀 他のトピックを見る**")
        oq_cols = st.columns(len(other_qs)) if len(other_qs) > 1 else [st.container()]
        for i, nq in enumerate(other_qs):
            with oq_cols[i]:
                if st.button(nq, key=f"next_q_hist_other_{i}", use_container_width=True):
                    st.session_state._suggest = nq
                    st.session_state._suggest_source = "suggest_next"
                    del st.session_state["next_questions"]
                    st.rerun()

if not st.session_state.messages and "_suggest" not in st.session_state:
    st.markdown("💡 **まずは、気になるボタンをタップしてみてね！**")
    suggest_question = None
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👶 子育て支援について", use_container_width=True):
            suggest_question = "子育て支援についてどのような議論がありましたか？"
        if st.button("💻 デジタル化・DXの進展について", use_container_width=True):
            suggest_question = "デジタル化・DXの進展についてはどのような議論がありましたか？"
    with col2:
        if st.button("🏫 小中学校の環境について", use_container_width=True):
            suggest_question = "小中学校の環境についてどのような議論がありましたか？"
        if st.button("👴 高齢者福祉について", use_container_width=True):
            suggest_question = "高齢者福祉についてはどのような議論がありましたか？"
    with st.expander("➕ もっと他のテーマを見る"):
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if st.button("🌊 防災・水害対策について", use_container_width=True):
                suggest_question = "防災や水害対策（ハザードマップや避難所など）についてどのような議論がありましたか？"
            if st.button("🌳 公園・みどりの充実について", use_container_width=True):
                suggest_question = "公園の整備やみどりの環境づくりについてどのような議論がありましたか？"
            if st.button("🚲 自転車・交通マナーについて", use_container_width=True):
                suggest_question = "自転車の安全対策や交通マナーについてはどのような議論がありましたか？"
        with ecol2:
            if st.button("🐕 ペット・動物愛護について", use_container_width=True):
                suggest_question = "犬や猫など、ペットの飼育環境や動物愛護についてはどのような議論がありましたか？"
            if st.button("🏢 起業・中小企業への支援について", use_container_width=True):
                suggest_question = "法人の設立支援や、中小企業への施策についてどのような議論がありましたか？"
            if st.button("💴 物価高騰・生活支援について", use_container_width=True):
                suggest_question = "物価高騰に対する生活支援や経済対策についてどのような議論がありましたか？"
    if suggest_question:
        st.session_state._suggest = suggest_question
        st.session_state._suggest_source = "suggest_initial"
        st.rerun()

chat_input_question = st.chat_input(
    "質問してみてね（例：予算審査で議論された主な項目は？　〇〇議員はどんな質問をしている？）"
)
if "_suggest" in st.session_state:
    question = st.session_state.pop("_suggest")
    source = st.session_state.pop("_suggest_source", "suggest_initial")
elif chat_input_question:
    question = chat_input_question
    source = "manual"
else:
    question = None
    source = "manual"

if question:
    if "next_questions" in st.session_state:
        del st.session_state["next_questions"]
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
                nq_match = re.search(r'\[NEXT_QUESTIONS\](.*?)\[/NEXT_QUESTIONS\]', answer, re.DOTALL)
                next_questions = []
                if nq_match:
                    block = nq_match.group(1)
                    next_questions = [
                        line.strip().lstrip('- ').strip()
                        for line in block.strip().split('\n')
                        if line.strip().startswith('-')
                    ]
                    clean_answer = re.sub(
                        r'\s*\[NEXT_QUESTIONS\].*?\[/NEXT_QUESTIONS\]', '',
                        answer, flags=re.DOTALL
                    ).strip()
                else:
                    clean_answer = answer
                st.markdown(clean_answer)
                save_log(question, clean_answer, source, cookies["user_id"], st.session_state.session_id)
                st.session_state.messages.append({"role": "assistant", "content": clean_answer})
                st.session_state._scroll_to_bottom = True
                if next_questions:
                    st.session_state.next_questions = next_questions
                    st.rerun()
            except Exception as e:
                st.error("申し訳ありません。回答の生成中にエラーが発生しました。時間をおいて再度お試しください。")

if st.session_state.pop("_scroll_to_bottom", False):
    components.html("""
    <script>
        setTimeout(function() {
            const main = window.parent.document.querySelector('[data-testid="stAppViewBlockContainer"]');
            if (main) main.scrollTop = main.scrollHeight;
        }, 200);
    </script>
    """, height=0)
