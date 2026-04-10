import os
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

# ================= 1. 環境變數設定 =================
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.environ.get('LINE_CHANNEL_SECRET')
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')

# 初始化 LINE Bot API
configuration = Configuration(access_token=LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ================= 2. AI 模型與 RAG 知識庫初始化 =================
# 推薦使用 llama-3-70b-instruct，對中文支援佳且邏輯強
llm = ChatNVIDIA(model="meta/llama3-8b-instruct", nvidia_api_key=NVIDIA_API_KEY)
embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", nvidia_api_key=NVIDIA_API_KEY)

vector_store = None

def initialize_rag():
    """啟動時讀取 data 資料夾內的檔案，並建立向量資料庫"""
    global vector_store
    data_dir = "./data"
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print("已建立 data 資料夾，請將 PDF/Word 講義放入此處。")
        return

    documents = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                documents.extend(loader.load())
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                documents.extend(loader.load())
        except Exception as e:
            print(f"讀取檔案 {filename} 時發生錯誤: {e}")

    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, embeddings)
        print("✅ 知識庫載入完成！機器人現在懂講義內容了。")
    else:
        print("⚠️ 警告：data 資料夾中沒有文件，機器人將以純對話模式啟動（無 RAG）。")

initialize_rag()

# ================= 3. 設定學長人設與提示詞 (Prompt) =================
system_prompt = (
    "你是一位在國家文官學院服務的貼心學長，請用親切、鼓勵且具備建設性的口吻，提供同學課業上的解決方案與建議，也會講笑話幫大家紓解壓力。\n"
    "【最高指令】：請務必、絕對要使用「繁體中文 (Traditional Chinese, zh-TW)」進行回覆，不管問題是什麼，都嚴禁使用簡體中文或其他語言。\n\n"
    "請優先根據以下提供的參考資料(Context)來回答問題。如果參考資料中沒有相關資訊，請用你豐富的常識回答，但務必保持溫暖的文官學長人設。\n\n"
    "【參考資料】：\n{context}\n\n"
    "【強制規定】：在每一次回答的最後，請務必換行並精準加上這句提醒文字：\n"
    "『⚠️ 溫馨提醒：以上內容為 AI 生成，僅供參考，實際課業規範與考試資訊，請務必以文官學院官方最新公告為準喔！』"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def get_ai_response(user_input):
    """處理使用者輸入並生成 AI 回覆"""
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    if vector_store:
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_input})
        return response["answer"]
    else:
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

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    user_message = event.message.text
    
    # ---------------------------------------------------------
    # 【新增功能：關鍵字防洗版機制】
    # 判斷訊息中是否包含「文官助理」
    if "文官助理" not in user_message:
        # 如果沒有關鍵字，直接結束函式，機器人不會已讀也不會回覆
        return
        
    # 如果有關鍵字，把「文官助理」這四個字清掉，以免干擾 AI 判斷問題
    clean_message = user_message.replace("文官助理", "").strip()
    
    # 如果同學只打了「文官助理」卻沒問問題，給予預設的引導對話
    if not clean_message:
         clean_message = "請用學長的人設簡短打個招呼，並問我有什麼需要幫忙的？"
    # ---------------------------------------------------------

    try:
        # 呼叫大腦處理文字 (傳入過濾後的乾淨訊息)
        ai_reply = get_ai_response(clean_message)
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

# ================= 5. 啟動伺服器 =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
