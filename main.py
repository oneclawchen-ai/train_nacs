import os
from apscheduler.schedulers.background import BackgroundScheduler
import pytz # 用來設定台灣時區
from langchain_core.messages import HumanMessage # 用來呼叫 LLM
from flask import Flask, request, abort
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi, ReplyMessageRequest, TextMessage
from linebot.v3.webhooks import MessageEvent, TextMessageContent

# LangChain 與 NVIDIA 模組
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = Flask(__name__)

# ---------------------------------------------------------
# 每日早安廣播排程設定
# ---------------------------------------------------------
def send_morning_greeting():
    try:
        prompt = """
        請以「貼心、幽默的文官學院學長學姊」身分，請務必、絕對要使用「繁體中文 (Traditional Chinese, zh-TW)」撰寫，寫一段約 10~50 字的早安勉勵語(包含中英文)。
        對象是正在受訓的文官學員。
        要求：
        1. 語氣要輕鬆、溫暖、充滿希望。
        2. 內容可以鼓勵他們面對今天的課程、專題壓力，或是簡單關心天氣。
        3. 每天的內容都要有新鮮感。
        4. 請加上適當的 Emoji 圖案來點綴。
        5. 不要出現「我是 AI」等字眼，要完全融入學長學姊角色。
        6. 不定時會送出勉勵的話，用中英文來送出勉勵的話。
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        greeting_text = response.content.strip()
        
        from linebot.v3.messaging import BroadcastRequest, TextMessage
        with ApiClient(configuration) as api_client:
            line_bot_api = MessagingApi(api_client)
            broadcast_request = BroadcastRequest(messages=[TextMessage(text=greeting_text)])
            line_bot_api.broadcast(broadcast_request)
        
        print(f"✅ 早安廣播已成功發送：\n{greeting_text}")
    except Exception as e:
        print(f"❌ 早安廣播發送失敗：{e}")

scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
scheduler.add_job(send_morning_greeting, 'cron', hour=7, minute=35)
scheduler.start()

# ================= 1. 環境變數設定 =================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ================= 2. AI 模型與 RAG 知識庫初始化 =================
# ⚠️ 將模型改回穩定的 NVIDIA 官方端點名稱，以免發生閃退
llm = ChatNVIDIA(model="openai/gpt-oss-20b", nvidia_api_key=NVIDIA_API_KEY, temperature=0.2, top_p=0.7)
embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", nvidia_api_key=NVIDIA_API_KEY,truncate="END")

vector_store = None

def initialize_rag():
    global vector_store
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("\n⚠️ [警告] 找不到 data 資料夾，已自動建立！(這代表 Render 雲端上目前沒有講義)")
        return

    documents = []
    files_count = 0
    
    print("\n📂 開始掃描 data 資料夾中的講義檔案...")
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        ext = filename.lower() # 防呆機制：確保 .PDF 也能讀取
        try:
            if ext.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
                files_count += 1
                print(f"  - 成功讀取: {filename}")
            elif ext.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                documents.extend(loader.load())
                files_count += 1
                print(f"  - 成功讀取: {filename}")
        except Exception as e:
            print(f"❌ 讀取檔案 {filename} 時發生錯誤: {e}")

    if files_count == 0:
        print("⚠️ [警告] data 資料夾是空的，沒有發現任何 PDF 或 Word 檔案！")

    if documents:
        # 【優化】加大 chunk_size 到 150，讓 AI 看到的上下文更完整，避免斷章取義
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        print(f"✅ 知識庫載入完成！共讀取 {files_count} 個檔案，切成了 {len(docs)} 個知識區塊。\n")
    else:
        print("⚠️ [警告] 機器人將以純對話模式啟動（無 RAG 輔助）。\n")

initialize_rag()

# ================= 3. 設定學長人設與提示詞 (Prompt) =================
system_prompt = (
    "你是一位在國家文官學院服務的貼心學長姊，請用親切、鼓勵且具備建設性的口吻像是同學好，提供同學課業上的解決方案與建議，也會講笑話幫大家紓解壓力。\n"
    "【最高指令】：請務必、絕對要使用「繁體中文 (Traditional Chinese, zh-TW)」進行回覆，不管問題是什麼，都嚴禁使用簡體中文或其他語言。\n\n"
    "【格式嚴格限制】：因為你的對話平台是 LINE 聊天室，它不支援任何 Markdown 格式！請「絕對不要」產出任何表格 (Table, 使用 | 符號的那種)。如果需要列點或比較資訊，請一律改用「條列式純文字」，例如使用「一、」、「(一)」或簡單的 Emoji 符號 (如 🔸、✅) 來分段排版。也不要使用 ### 或 ** 等標記符號。\n\n"
    "請優先根據以下提供的參考資料(Context)來回答問題。如果參考資料中沒有相關資訊，就回答這不是我專業請洽詢輔導員，千萬不要亂回答，但務必保持溫暖的文官學長人設，最後都會送上一句祝福的話。\n\n"
    "【參考資料】：\n{context}\n\n"
    "【強制規定】：在每一次回答的最後，請務必換行並精準加上這句提醒文字：\n"
    "『⚠️ 溫馨提醒：以上內容為 AI 生成，僅供參考，實際課業規範與考試資訊，請務必以文官學院官方最新公告為準喔！』"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def get_ai_response(user_input):
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    if vector_store:
        # 【優化】讓檢索器找最相關的 4 塊資料 (k=4)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # --- 【新增日誌探照燈】將 RAG 抓到的資料印在 Render 的 Logs 讓我們看 ---
        try:
            retrieved_docs = retriever.invoke(user_input)
            print(f"\n🔍 [RAG 檢索測試] 同學提問：「{user_input}」")
            if not retrieved_docs:
                print("   ⚠️ 糟糕！知識庫裡完全找不到相關內容。")
            else:
                print(f"   📦 RAG 共抓出 {len(retrieved_docs)} 塊資料餵給 AI：")
                for i, doc in enumerate(retrieved_docs):
                    source = doc.metadata.get('source', '未知檔案')
                    print(f"   📄 [來源 {i+1} : {os.path.basename(source)}] 預覽: {doc.page_content[:60]}...")
            print("-" * 50)
        except Exception as e:
            print(f"❌ 檢索過程發生錯誤: {e}")
        # ---------------------------------------------------------
        
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        return response["answer"]
    else:
        print(f"⚠️ [無知識庫模式] 同學提問：「{user_input}」")
        response = document_chain.invoke({"input": user_input, "context": []})
        return response

# ================= 4. LINE Webhook 路由設定 =================
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("無效的簽章！請檢查 LINE_CHANNEL_SECRET。")
        abort(400)
    return 'OK'

# ================= 4. LINE Webhook 路由設定 =================
# (原本的 /callback 路由保留在上面)

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    
    # 判斷訊息中是否包含「文官助理」
    if "文官助理" not in user_message:
        # 如果沒有關鍵字，直接結束函式，機器人不會已讀也不會回覆
        return
        
    # 如果有關鍵字，把「文官助理」這四個字清掉，以免干擾 AI 判斷問題
    clean_message = user_message.replace("文官助理", "").strip()
    
    # 如果同學只打了「文官助理」卻沒問問題，給予預設的引導對話
    if not clean_message:
         clean_message = "請用學長的人設簡短打個招呼，並問我有什麼需要幫忙的？"

    try:
        # 呼叫大腦處理文字
        ai_reply = get_ai_response(clean_message)
        
        # ---------------------------------------------------------
        # 🧹 【字串淨水器】：過濾掉 AI 喜歡加的 Markdown 符號
        # 把 "###" 和 "**" 替換成空字串 (也就是刪除)
        ai_reply = ai_reply.replace("###", "")
        ai_reply = ai_reply.replace("**", "")
        ai_reply = ai_reply.strip()
        # ---------------------------------------------------------
        
    except Exception as e:
        ai_reply = f"不好意思同學，學長的大腦暫時連不上線啦 (系統錯誤: {str(e)})，請稍後再試！"

    # 將結果回傳給 LINE 使用者
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message_with_http_info(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=ai_reply)]
            )
        )

@app.route("/", methods=['GET'])
def hello():
    return "Hello, the bot is running!"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
