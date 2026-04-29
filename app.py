import os
import re
import json
import uuid
import queue
import threading
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted

st.set_page_config(
    page_title="教えて！えどがわ区議会AI",
    page_icon="🦉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    [data-testid="stDecoration"] {
        visibility: hidden !important;
    }
    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    footer {visibility: hidden;}
    /* サイドバー展開ボタンを強制表示＆最前面へ */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: block !important;
        z-index: 999999 !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
[data-testid="stChatMessage"] h1 { font-size: 1.4rem; }
[data-testid="stChatMessage"] h2 { font-size: 1.2rem; }
[data-testid="stChatMessage"] h3 { font-size: 1.05rem; }
[data-testid="stChatMessage"] h4 { font-size: 1.0rem; }
@keyframes _fade1{0%,44%,100%{opacity:1}50%,94%{opacity:0}}
@keyframes _fade2{0%,44%,100%{opacity:0}50%,94%{opacity:1}}
/* pills内のテキスト見切れ防止 */
[data-testid="stPills"] button,
[data-testid="stPills"] button p,
[data-testid="stPills"] button span {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    word-break: break-all !important;
    height: auto !important;
    min-height: 2rem !important;
    line-height: 1.4 !important;
}
/* チャット入力欄に薄い影をつけて浮遊感を出す */
[data-testid="stChatInput"] {
    box-shadow: 0 -4px 20px rgba(0,0,0,0.04) !important;
    border: none !important;
}
/* アバターを丸く */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {
    border-radius: 50% !important;
}
/* チャットバブルをテーマカラー（極淡い青）で装飾 */
[data-testid="stChatMessage"] {
    background-color: #F4F8FF !important;
    border: 1px solid #C5D7FB !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    margin-bottom: 0.75rem !important;
}
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


KNOWN_COMMITTEES = [
    "生活振興環境委員会", "健康推進・熟年者支援特別委員会",
    "災害対策・街づくり推進特別委員会", "行財政改革・ＳＤＧｓ推進特別委員会",
    "子育て・教育委員会", "総務委員会", "建設委員会", "予算特別委員会",
    "福祉健康委員会", "文教委員会",
]


@st.cache_resource(show_spinner=False)
def get_vectorstore(api_key: str):
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
    return vectorstore


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def _similarity_search_with_retry(vectorstore, *args, **kwargs):
    return vectorstore.similarity_search(*args, **kwargs)


@st.cache_data(ttl=3600, show_spinner=False)
def search_docs(_vectorstore, question: str, k: int = 20) -> list:
    """委員会名が含まれる場合はメタデータフィルター付き検索、それ以外は通常検索。"""
    from google.cloud.firestore_v1.base_query import FieldFilter
    matched = [c for c in KNOWN_COMMITTEES if c in question]
    if matched:
        # 複数委員会の場合は1委員会あたりのkを制限（コンテキスト膨張防止）
        per_k = max(5, k // max(len(matched), 1))
        all_docs = []
        for committee in matched:
            # 通常クエリ
            filtered = _similarity_search_with_retry(
                _vectorstore, question, k=per_k,
                filters=FieldFilter("metadata.committee_name", "==", committee),
            )
            # 政策テーマ特化クエリ（事務的な日程・手続きを回避）
            extra = _similarity_search_with_retry(
                _vectorstore,
                f"{committee} 政策 審議 報告 区民 予算 条例 事業",
                k=per_k,
                filters=FieldFilter("metadata.committee_name", "==", committee),
            )
            all_docs.extend(filtered + extra)
        return list({d.page_content: d for d in all_docs}.values())
    return _similarity_search_with_retry(_vectorstore, question, k=k)


cookies = CookieManager()
if not cookies.ready():
    st.stop()
if "user_id" not in cookies:
    cookies["user_id"] = str(uuid.uuid4())
    cookies.save()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

vectorstore = get_vectorstore(api_key)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは江戸川区議会の議事録を読み解き、区民に分かりやすく立体的に解説する、優秀なWebメディア編集長（えどがわ議会AI）です。
知識ベースには、令和8年度予算特別委員会（最新）と、令和5年度以降の各種会議の議事録が含まれています。
以下の議事録の抜粋を参考にして、質問に日本語で正確に答えてください。

【回答の構成とルール（厳守）】
冒頭の「承知いたしました」といった挨拶や自己紹介は一切不要です。質問の意図を汲み取り、以下の1〜4の構成で出力してください。

1. 🔥 30秒でわかる！今回のハイライト
質問に対する結論や、一番の「見どころ」を3〜4行でキャッチーに要約してください。

2. 🔍 注目のポイント（※質問のタイプに合わせて柔軟に構成を変更すること！）
ユーザーの質問タイプに合わせて、以下のいずれか最も適したアプローチで解説を展開してください。見出しもWeb記事風に魅力的なものにします。

- パターンA【特定のテーマ・施策についての質問の場合】
  見出し例：「⚖️ 有料化のメリットと、浮き彫りになった〇〇の課題」
  単純な「賛成・反対」の二極化にこだわらず、「期待される効果（推進側の意見）」と「懸念される課題（慎重派の意見やリスク）」という切り口で整理してください。
  ⚠️【重要】一人の発言者がメリットと懸念の両方を語っている場合は、発言を無理に分割せず「〇〇委員は必要性を理解しつつも、△△の課題を指摘した」と自然な文脈でまとめてください。議員の指摘に対して、区（役所側）がどう答弁したかという「質疑応答のキャッチボール」が分かるように記述してください。
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
- 【時制の明示（厳守）】ユーザーの質問に「今年度」「令和8年度」「新規」「新しく」「今回」など時期を特定する言葉が含まれる場合は、検索結果の中から「指定された年度（または最新）の議論」を最優先して抽出すること。過去の事業・議論に言及する際は「令和〇年の〇〇委員会では〜」と必ず過去であることを明記し、現在・今後の動向と混同させないこと。指定年度の新規情報が検索結果に見当たらない場合は、推測や類推で補わず「知識ベースに該当年度の新規事業に関する議論は見当たりませんでした」と正直に記載すること。
- 議員の名前（〇〇委員）と具体的な数字は必ず盛り込んでください。
- 【定量データの強制（厳守）】政策の効果・課題・予算規模を説明するときは、検索結果に含まれる「金額（○億円・○万円）」「対象人数・件数（○人・○件・○校中○校）」「割合（○%）」などの定量的な根拠（ファクト）を必ず探し出して積極的に引用すること。検索結果に数字が見当たらない場合のみ、その旨を断ったうえで定性的な説明を行うこと。数字による裏付けなく印象論に終始する文章は禁止。
- お役所言葉の単調な要約は避け、読者の目を引く魅力的な文章（「〜と鋭く指摘！」「果たして〜でしょうか？」など）にしてください。
- 【多様性の確保（厳守）】回答を作成する際は、可能な限り「複数の異なる議員」の発言をピックアップし、多角的な視点を含めてください。特定の議員1人の発言だけで記事を構成するのは避けてください。
- 万が一、読み込んだ議事録の中に1人の議員の発言しか含まれていない場合は、「今回の検索範囲では、主に〇〇委員から集的な質問がありました」と事実を明記し、議論の広がりが限定的であることを読者に伝えてください。
- ❌禁止：「推進・賛成する意見：」「慎重・反対する意見：」「現状と課題：」といった単調な箇条書きのラベル付けは絶対に行かないでください。代わりに、具体的な内容を要約した魅力的な小見出し（###）を作成して整理してください。
- 【議案名・条例名の明示（厳守）】検索結果のチャンクに「【議題】」タグが含まれる場合は、その議題名（例：「〇〇条例改正案の審査」「令和〇年度一般会計補正予算の審査」など）を必ず本文中に引用すること。議題名なしに「〇〇について議論がありました」と抽象的に述べることは禁止する。
- 【発言コンテキストの活用（推奨）】チャンクに「【直前の発言】」タグが含まれる場合は、その文脈（誰が何を言った直後の発言か）を考慮して発言の意図を正確に解釈すること。特に「賛成します」「了解しました」などの短い発言は、直前の発言を参照して何に同意したかを明記すること。

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


def stream_and_extract(chain, inputs):
    """chain.stream() から [NEXT_QUESTIONS] ブロックを除去しながら yield する。
    ブロック内容は st.session_state._streamed_nq_raw に格納される。"""
    buffer = ""
    in_block = False
    block_buf = ""
    st.session_state._streamed_nq_raw = None

    for chunk in chain.stream(inputs):
        if not isinstance(chunk, str):
            chunk = chunk.content if hasattr(chunk, "content") else str(chunk)

        if not in_block:
            buffer += chunk
            if "[NEXT_QUESTIONS]" in buffer:
                before, rest = buffer.split("[NEXT_QUESTIONS]", 1)
                if before:
                    yield before
                # rest をそのまま block_buf に移す（buffer でなく）
                block_buf = rest
                buffer = ""
                in_block = True
                # 同一チャンク内に [/NEXT_QUESTIONS] が既にある場合
                if "[/NEXT_QUESTIONS]" in block_buf:
                    content, _ = block_buf.split("[/NEXT_QUESTIONS]", 1)
                    st.session_state._streamed_nq_raw = content
                    return
            else:
                safe, tail = buffer.rsplit("\n", 1) if "\n" in buffer else ("", buffer)
                if safe:
                    yield safe + "\n"
                    buffer = tail
        else:
            block_buf += chunk
            if "[/NEXT_QUESTIONS]" in block_buf:
                content, _ = block_buf.split("[/NEXT_QUESTIONS]", 1)
                st.session_state._streamed_nq_raw = content
                return

    # ストリーム終了時の後処理
    if not in_block and buffer.strip():
        yield buffer
    # in_block のまま終了した場合（モデルが閉じタグを出力しなかった）は questions なしで終了


_SPINNER_HTML = (
    '<span style="display:inline-block;width:13px;height:13px;'
    'border:2px solid #ccc;border-top-color:#888;border-radius:50%;'
    'animation:_spin 0.8s linear infinite;vertical-align:middle;margin-left:6px">'
    '</span>'
    '<style>@keyframes _spin{to{transform:rotate(360deg)}}</style>'
)


def _rotating_status_html() -> str:
    return (
        '<div style="position:relative;height:1.6em;font-size:1rem">'
        f'<span style="position:absolute;animation:_fade1 6s ease-in-out infinite">'
        f'✍️ AIが原稿を書いています...{_SPINNER_HTML}</span>'
        f'<span style="position:absolute;animation:_fade2 6s ease-in-out infinite">'
        f'⏳ もう少しお待ちください...{_SPINNER_HTML}</span>'
        '</div>'
    )


def _stream_clear_status(gen, status_placeholder):
    """最初のチャンクが来た瞬間にステータスプレースホルダーを消去するラッパー。"""
    first = True
    for chunk in gen:
        if first:
            status_placeholder.empty()
            first = False
        yield chunk


_SPEAKER_BADGE = {
    "理事者": ("background:#f3e8ff;color:#7e22ce", "🏛️"),
    "委員長": ("background:#fff7ed;color:#c2410c", "⚖️"),
    "議長":   ("background:#fffbeb;color:#b45309", "🔔"),
    "議員":   ("background:#f0fdf4;color:#15803d", "💬"),
}

_REIWA_BASE = 2018  # 令和N年 = 2018 + N


def _parse_doc(doc) -> dict:
    """page_content のテキストから発言メタ情報を抽出する。"""
    c = doc.page_content
    def _get(pattern):
        m = re.search(pattern, c)
        return m.group(1).strip() if m else ""

    date_jp  = _get(r"開催日：(.+)")
    speaker  = _get(r"発言者：([^（\n]+)")
    stype    = _get(r"属性：([^）\n]+)")
    committee = _get(r"会議名：[^（\n]*（([^）\n]+)）")
    speech_m = re.search(r"発言内容：\n(.*?)(?:\n---|$)", c, re.DOTALL)
    speech   = speech_m.group(1).strip() if speech_m else c
    agenda_title = _get(r"【議題】(.+)")

    # 令和→西暦変換（令和N年 → 2018+N）
    reiwa_m = re.search(r"令和(\d+)年", date_jp)
    year = (_REIWA_BASE + int(reiwa_m.group(1))) if reiwa_m else None
    # 西暦がそのまま書かれている場合のフォールバック
    if year is None:
        wy = re.search(r"(\d{4})年", date_jp)
        year = int(wy.group(1)) if wy else None

    return {
        "date_jp": date_jp,
        "speaker": speaker or "不明",
        "stype": stype,
        "committee": committee,
        "speech": speech,
        "year": year,
        "agenda_title": agenda_title,
    }


def _render_source_cards(docs: list):
    """検索結果docsをカード形式で表示する。"""
    seen: set[str] = set()
    unique_docs = []
    for d in docs:
        key = d.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    parsed = [_parse_doc(d) for d in unique_docs]
    st.caption(f"参照した発言：{len(parsed)} 件")

    for p in parsed:
        badge_style, badge_icon = _SPEAKER_BADGE.get(
            p["stype"], ("background:#eff6ff;color:#1d4ed8", "📋")
        )
        with st.container(border=True):
            header_parts = []
            if p.get("agenda_title"):
                header_parts.append(
                    f'<span style="font-size:0.75rem;padding:2px 8px;border-radius:4px;'
                    f'background:#e0f2fe;color:#0369a1;font-weight:600;">📋 {p["agenda_title"]}</span>'
                )
            header_parts.append(
                f'<span style="font-size:0.75rem;padding:2px 8px;border-radius:4px;'
                f'font-weight:600;{badge_style}">{badge_icon} {p["stype"]}</span>'
            )
            st.markdown(" ".join(header_parts), unsafe_allow_html=True)
            st.markdown(f'**{p["speaker"]}**　{p["committee"]}　{p["date_jp"]}')
            st.caption(p["speech"][:400] + ("…" if len(p["speech"]) > 400 else ""))


@st.cache_data(show_spinner=False)
def get_recent_meetings():
    json_path = os.path.join(os.path.dirname(__file__), "recent_meetings.json")
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


recent_prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは江戸川区議会の議事録を区民にわかりやすく伝える編集長（えどがわ議会AI）です。
以下の議事録の抜粋をもとに、複数の会議それぞれの概要を整理してください。

【出力ルール（厳守）】
- 各会議について `## 📅 [会議名]（[日付]）` の見出しで区切ること
- 各会議の内容は以下の4項目の構成で記述すること：
  1. **🗂️ 主な議題**：箇条書き2〜4件（議題名だけでなく、何が問題になっているか・何を目的としているかを1文添えること）
  2. **💬 議論のポイント**：賛成・反対・懸念など議員の意見の対立や、区側の答弁の要点を2〜4行で具体的に説明すること。数字や固有名詞があれば積極的に盛り込むこと
  3. **🗣️ 注目の発言**：議事録から印象的な発言・質問を1〜2件、カギカッコで引用し、発言者名（役職含む）を添えること。引用できる発言がない場合はこの項目は省略してよい
  4. **📌 今後の注目**：この会議を受けて次に何が起きるか・何が未解決かを1〜2行でまとめること
- 全会議の後に横断的な「まとめ」は不要。
- `#`（H1）は絶対使用しない。
- 冒頭の挨拶・自己紹介は不要。すぐに各会議の内容を出力すること。

【次の質問候補（必須）】
回答の最後に5つの質問候補を以下の形式で出力してください。
候補は全て、上記の会議の内容を深掘りするものにすること（他分野のトピックは不要）。
[NEXT_QUESTIONS]
- 質問候補1
- 質問候補2
- 質問候補3
- 質問候補4
- 質問候補5
[/NEXT_QUESTIONS]

【議事録の抜粋】
{context}"""),
    ("human", "{question}"),
])


_log_queue: queue.Queue = queue.Queue(maxsize=100)


def _log_worker():
    while True:
        try:
            item = _log_queue.get(timeout=1)
        except queue.Empty:
            continue
        try:
            question, answer, source, user_id, session_id = item
            credentials = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(credentials)
            sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
            worksheet = sh.get_worksheet(0)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row([now, user_id, session_id, source, question, answer])
        except Exception:
            print("Logging error: failed to write to spreadsheet")


threading.Thread(target=_log_worker, daemon=True).start()


def save_log(question, answer, source="manual", user_id="", session_id=""):
    try:
        _log_queue.put_nowait((question, answer, source, user_id, session_id))
    except queue.Full:
        print("Logging error: queue full")


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    # 1. コンセプト説明カード
    st.markdown(
        """
        <div style='background-color: #F4F8FF; border: 1px solid #C5D7FB; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;'>
            <div style='font-size: 1rem; font-weight: 700; color: #000060; margin-bottom: 0.5rem;'>えどがわ議会AIとは？</div>
            <div style='font-size: 0.85rem; color: #475569; line-height: 1.6;'>
                「議事録を読む時間がない…」を解決する区民のための検索アシスタントです。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. アプリの機能リスト
    st.markdown(
        """
        <div style='padding-left: 0.2rem; margin-bottom: 2rem;'>
            <div style='font-size: 0.85rem; color: #475569; margin-bottom: 0.6rem; display: flex; align-items: flex-start;'>
                <span style='color: #3460FB; font-weight: 900; margin-right: 0.6rem; font-size: 0.9rem;'>✓</span> <span style='line-height: 1.4;'>令和5年からの議論を網羅</span>
            </div>
            <div style='font-size: 0.85rem; color: #475569; margin-bottom: 0.6rem; display: flex; align-items: flex-start;'>
                <span style='color: #3460FB; font-weight: 900; margin-right: 0.6rem; font-size: 0.9rem;'>✓</span> <span style='line-height: 1.4;'>AIが要点をわかりやすく解説</span>
            </div>
            <div style='font-size: 0.85rem; color: #475569; margin-bottom: 0.6rem; display: flex; align-items: flex-start;'>
                <span style='color: #3460FB; font-weight: 900; margin-right: 0.6rem; font-size: 0.9rem;'>✓</span> <span style='line-height: 1.4;'>出典元の生の発言も確認可能</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 3. メタ情報（フッター風）
    st.markdown(
        """
        <div style='border-top: 1px solid #e2e8f0; padding-top: 1.2rem; margin-top: 2rem;'>
            <div style='font-size: 0.75rem; color: #64748b; margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem;'>
                <strong>作成者：</strong>
                <a href="https://x.com/edogawa_aki" target="_blank" style='color: #3460FB; text-decoration: none;'>あき@データで見る江戸川区</a>
            </div>
            <div style='font-size: 0.7rem; color: #94a3b8; line-height: 1.6;'>
                <span style='font-weight: bold;'>🔒 ログの収集について</span><br>
                入力内容はアプリ改善のため匿名で記録されます。個人情報は入力しないでください。
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

AVATARS = {"user": "👤", "assistant": "🦉"}

_SUGGEST_MAP = {
    "📅 最近の話題": "__RECENT__",
    "💴 令和8年度予算": "令和8年度予算特別委員会で審議された主な項目と議論のポイントを教えてください。",
    "👶 子育て支援": "子育て支援についてどのような議論がありましたか？",
    "👴 高齢者福祉": "高齢者福祉についてはどのような議論がありましたか？",
    "🏫 小中学校": "小中学校の環境についてどのような議論がありましたか？",
    "💴 物価高騰対策": "物価高騰に対する生活支援や経済対策についてどのような議論がありましたか？",
    "🐕 ペット・動物愛護": "犬や猫など、ペットの飼育環境や動物愛護についてはどのような議論がありましたか？",
    "🌊 防災・水害対策": "防災や水害対策（ハザードマップや避難所など）についてどのような議論がありましたか？",
    "🌳 公園・みどり": "公園の整備やみどりの環境づくりについてどのような議論がありましたか？",
    "🚲 自転車・交通": "自転車の安全対策や交通マナーについてはどのような議論がありましたか？",
    "🏢 中小企業支援": "法人の設立支援や、中小企業への施策についてどのような議論がありましたか？",
    "💻 デジタル化・DX": "デジタル化・DXの進展についてはどのような議論がありましたか？",
}

if not st.session_state.messages and "_suggest" not in st.session_state:
    # --- Empty State（初回アクセス）---
    st.markdown(
        """
        <div style='margin-bottom: 2rem; margin-top: 1rem;'>
            <h1 style='text-align: left; color: #000060; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.2rem; letter-spacing: 0.05em;'>教えて！えどがわ議会AI</h1>
            <p style='text-align: left; color: #64748b; font-size: 0.9rem; margin-top: 0;'>江戸川区議会の過去の議論をAIがやさしく解説します。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.chat_message("assistant", avatar="🦉"):
        st.markdown(
            "こんにちは！江戸川区議会に関する過去の議論をわかりやすく解説します。"
            "気になるトピックを選んでみてください。"
        )
    with st.container(border=True):
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.1em; margin-bottom: 1.2rem; margin-top: 0;'>注目・トレンド</div>",
            unsafe_allow_html=True,
        )
        sel_trend = st.pills(
            "trend",
            ["📅 最近の話題", "💴 令和8年度予算"],
            key="pills_trend",
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.1em; margin-bottom: 1.2rem; margin-top: 1.5rem;'>暮らし・福祉</div>",
            unsafe_allow_html=True,
        )
        sel_living = st.pills(
            "living",
            ["👶 子育て支援", "👴 高齢者福祉", "🏫 小中学校", "💴 物価高騰対策", "🐕 ペット・動物愛護"],
            key="pills_living",
            label_visibility="collapsed",
        )
        st.markdown(
            "<div style='font-size: 0.75rem; font-weight: 600; color: #64748b; letter-spacing: 0.1em; margin-bottom: 1.2rem; margin-top: 1.5rem;'>地域・インフラ</div>",
            unsafe_allow_html=True,
        )
        sel_infra = st.pills(
            "infra",
            ["🌊 防災・水害対策", "🌳 公園・みどり", "🚲 自転車・交通", "🏢 中小企業支援", "💻 デジタル化・DX"],
            key="pills_infra",
            label_visibility="collapsed",
        )
    for sel in [sel_trend, sel_living, sel_infra]:
        if sel:
            mapped = _SUGGEST_MAP.get(sel, sel)
            if mapped == "__RECENT__":
                st.session_state._recent_mode = True
            else:
                st.session_state._suggest = mapped
                st.session_state._suggest_source = "suggest_initial"
            st.rerun()
else:
    # --- Chat Mode（会話中）---
    for i, message in enumerate(st.session_state.messages):
        is_last_ai_with_docs = (
            message["role"] == "assistant"
            and i == len(st.session_state.messages) - 1
            and st.session_state.get("_last_docs")
            and st.session_state.get("_last_source") != "suggest_recent"
        )
        if is_last_ai_with_docs:
            with st.chat_message("assistant", avatar="🦉"):
                tab1, tab2 = st.tabs(["🦉 AI解説", "📄 元の議事録"])
                with tab1:
                    with st.container(height=400, border=False):
                        st.markdown(message["content"])
                with tab2:
                    with st.container(height=400, border=False):
                        _render_source_cards(st.session_state._last_docs)
        else:
            with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
                st.markdown(message["content"])

    if (st.session_state.get("next_questions")
            and st.session_state.messages
            and st.session_state.messages[-1]["role"] == "assistant"):
        is_recent_mode = st.session_state.get("_last_source") == "suggest_recent"
        all_qs = st.session_state.next_questions
        chosen = None

        if is_recent_mode:
            st.markdown("**🔍 この会議を深掘りする**")
            for i, nq in enumerate(all_qs):
                if st.button(nq, key=f"nq_{len(st.session_state.messages)}_{i}", use_container_width=True):
                    chosen = nq
                    del st.session_state["next_questions"]
                    if "_last_source" in st.session_state:
                        del st.session_state["_last_source"]
        else:
            deep_qs = all_qs[:3]
            other_qs = all_qs[3:]
            st.markdown("**💡 続けてこんな質問はいかがですか？**")
            for i, nq in enumerate(deep_qs):
                if st.button(nq, key=f"nq_{len(st.session_state.messages)}_{i}", use_container_width=True):
                    chosen = nq
                    del st.session_state["next_questions"]
            if other_qs:
                st.markdown("**🔀 他のトピックを見る**")
                for i, nq in enumerate(other_qs):
                    if st.button(nq, key=f"nq_{len(st.session_state.messages)}_o{i}", use_container_width=True):
                        chosen = nq
                        del st.session_state["next_questions"]

        if chosen:
            st.session_state._suggest = chosen
            st.session_state._suggest_source = "suggest_next"
            st.rerun()

chat_input_question = st.chat_input(
    "質問してみてね（例：予算審査で議論された主な項目は？　〇〇議員はどんな質問をしている？）"
)

# 直近会議モードの処理
recent_mode = st.session_state.pop("_recent_mode", False)

if "_suggest" in st.session_state:
    question = st.session_state.pop("_suggest")
    source = st.session_state.pop("_suggest_source", "suggest_initial")
elif recent_mode:
    meetings = get_recent_meetings()
    meeting_list_str = "\n".join(
        f"- {m['date_jp']} {m['meeting_name']}"
        for m in meetings
    )
    question = f"以下の会議それぞれについて、主な議題と議論の概要を教えてください：\n{meeting_list_str}"
    source = "suggest_recent"
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
        try:
            status = st.empty()
            if source == "suggest_recent":
                # 直近会議モード: 会議ごとにsearch_docsを呼んでcontextを結合
                st.session_state._last_docs = None
                status.markdown(f"📅 最近の会議を読み込んでいます...{_SPINNER_HTML}", unsafe_allow_html=True)
                all_docs = []
                for m in meetings:
                    q = f"{m['date_jp']} {m['meeting_name']} 議題 議論"
                    all_docs.extend(search_docs(vectorstore, q, k=5))
                context = format_docs(all_docs)
                chain = recent_prompt | llm | StrOutputParser()
                status.markdown(_rotating_status_html(), unsafe_allow_html=True)
                inputs = {"context": context, "question": question}
                displayed = st.write_stream(_stream_clear_status(stream_and_extract(chain, inputs), status))
            else:
                status.markdown(f"🔍 議事録を読み込んでいます...{_SPINNER_HTML}", unsafe_allow_html=True)
                docs = search_docs(vectorstore, question)
                st.session_state._last_docs = docs
                context = format_docs(docs)
                print(f"[DEBUG] context先頭300文字: {context[:300]}", flush=True)
                chain = prompt | llm | StrOutputParser()
                status.markdown(_rotating_status_html(), unsafe_allow_html=True)
                inputs = {"context": context, "chat_history": chat_history, "question": question}
                displayed = st.write_stream(_stream_clear_status(stream_and_extract(chain, inputs), status))

            clean_answer = displayed
            nq_raw = st.session_state.pop("_streamed_nq_raw", None)
            next_questions = []
            if nq_raw:
                next_questions = [
                    line.strip().lstrip("- ").strip()
                    for line in nq_raw.strip().split("\n")
                    if line.strip().startswith("-")
                ]
            save_log(question, clean_answer, source, cookies["user_id"], st.session_state.session_id)
            st.session_state.messages.append({"role": "assistant", "content": clean_answer})
            st.session_state._scroll_to_bottom = True
            if next_questions:
                st.session_state.next_questions = next_questions
                st.session_state._last_source = source
                st.rerun()
        except Exception as e:
            st.error("申し訳ありません。回答の生成中にエラーが発生しました。時間をおいて再度お試しください。")

if st.session_state.pop("_scroll_to_bottom", False):
    components.html("""
    <script>
        setTimeout(function() {
            const messages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length >= 2) {
                // 最後から2番目 = 新しいユーザーメッセージ（最後はAIの回答）
                messages[messages.length - 2].scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);
    </script>
    """, height=0)

components.html(
    """
    <script>
    window.parent.document.documentElement.lang = 'ja';
    </script>
    """,
    height=0,
)
