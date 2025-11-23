import os
import json
import hashlib
import re
import zipfile
import time
import random
from datetime import datetime
from collections import defaultdict
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from google.cloud import texttospeech
from google.oauth2 import service_account
import asyncio
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Environment variables - SAFE FOR GITHUB/RAILWAY
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

# Configuration
class Config:
    MAX_TOPIC_LENGTH = 100
    MAX_VOCAB_ITEMS = 15
    TTS_TIMEOUT = 30  # seconds
    API_RETRY_ATTEMPTS = 3
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for Telegram
    
    # Chirp3 voices for main text and opinion texts (random selection)
    CHIRP3_VOICES = [
        "cmn-CN-Chirp3-HD-Aoede",
        "cmn-CN-Chirp3-HD-Erinome",
        "cmn-CN-Chirp3-HD-Laomedeia",
        "cmn-CN-Chirp3-HD-Leda",
        "cmn-CN-Chirp3-HD-Puck",
        "cmn-CN-Chirp3-HD-Schedar"
    ]
    
    # Fixed voice for Anki vocabulary cards
    ANKI_VOICE = "cmn-CN-Chirp3-HD-Leda"

config = Config()

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# Rate Limiter
class RateLimiter:
    def __init__(self, max_requests=5, window=3600):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests[user_id]
        user_requests[:] = [req_time for req_time in user_requests if now - req_time < self.window]
        if len(user_requests) >= self.max_requests:
            return False
        user_requests.append(now)
        return True
    
    def get_reset_time(self, user_id):
        if not self.requests[user_id]:
            return 0
        oldest_request = min(self.requests[user_id])
        reset_time = oldest_request + self.window - time.time()
        return max(0, int(reset_time))

rate_limiter = RateLimiter(
    max_requests=config.RATE_LIMIT_REQUESTS,
    window=config.RATE_LIMIT_WINDOW
)

def get_google_tts_client():
    if GOOGLE_CREDENTIALS_JSON:
        credentials_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            credentials_dict,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return texttospeech.TextToSpeechClient(credentials=credentials)
    else:
        return texttospeech.TextToSpeechClient()

def validate_topic(topic):
    topic = re.sub(r'\s+', ' ', topic.strip())
    if re.search(r'[<>"|&;`$()]', topic):
        raise ValueError("Topic contains invalid characters")
    inappropriate_patterns = [
        r'\b(porn|sex|violence|hate|kill|death)\b',
        r'\b(暴力|色情|仇恨|歧视|杀|死)\b',
    ]
    for pattern in inappropriate_patterns:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ValueError("Topic contains inappropriate content")
    if len(topic) > config.MAX_TOPIC_LENGTH:
        topic = topic[:config.MAX_TOPIC_LENGTH]
    if not topic:
        raise ValueError("Topic cannot be empty")
    return topic

def split_text_into_sentences(text, max_length=150):
    sentences = re.split(r'([。！？；])', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        if i+1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
        else:
            result.append(sentences[i])
    final_result = []
    for sentence in result:
        if len(sentence) > max_length:
            parts = re.split(r'([，、])', sentence)
            temp = ""
            for part in parts:
                if len(temp + part) > max_length and temp:
                    final_result.append(temp)
                    temp = part
                else:
                    temp += part
            if temp:
                final_result.append(temp)
        else:
            final_result.append(sentence)
    return [s.strip() for s in final_result if s.strip()]

@retry(
    stop=stop_after_attempt(3),  # Increased from 2 to 3
    wait=wait_exponential(multiplier=1, min=3, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def generate_tts_chirp3_sync(text, voice_name=None, speaking_rate=1.0):
    """
    Generate Chinese TTS audio using Google Cloud Chirp3 (sync version)
    """
    client = get_google_tts_client()
    
    if voice_name is None:
        voice_name = random.choice(config.CHIRP3_VOICES)

    speaking_rate = max(0.25, min(2.0, speaking_rate))
    
    text_preview = text[:50].replace('\n', ' ').replace('\r', ' ')
    print(f"[TTS] 🎙️ Attempting TTS | Voice: {voice_name} | Speed: {speaking_rate} | Text: '{text_preview}...'")

    try:
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="cmn-CN",
            name=voice_name
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )
        
        # Enforce timeout to prevent hanging
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
            timeout=config.TTS_TIMEOUT - 5  # leave buffer for network
        )
        
        if not response.audio_content:
            raise ValueError("TTS returned empty audio_content")
        
        print(f"[TTS] ✅ SUCCESS | Voice: {voice_name} | Generated {len(response.audio_content)} bytes")
        return response.audio_content

    except Exception as e:
        print(f"[TTS] ❌ FAILED | Voice: {voice_name} | Error: {type(e).__name__}: {str(e)}")
        print(f"[TTS]    Text was: '{text_preview}...'")
        raise

async def generate_tts_async(text, voice_name=None, speaking_rate=1.0):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_chirp3_sync, text, voice_name, speaking_rate)

def safe_filename(filename):
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    filename = os.path.basename(filename)
    filename = filename[:100]
    return filename.strip('_')

def validate_deepseek_response(content):
    required_keys = ["main_text", "vocabulary", "opinion_texts", "discussion_questions"]
    if not all(k in content for k in required_keys):
        missing = [k for k in required_keys if k not in content]
        raise ValueError(f"Missing required keys in DeepSeek response: {missing}")
    if not isinstance(content['vocabulary'], list):
        raise ValueError("vocabulary must be a list")
    if len(content['vocabulary']) > config.MAX_VOCAB_ITEMS:
        content['vocabulary'] = content['vocabulary'][:config.MAX_VOCAB_ITEMS]
    for item in content['vocabulary']:
        if not all(k in item for k in ['english', 'chinese', 'pinyin']):
            raise ValueError("Each vocabulary item must have 'english', 'chinese', 'pinyin'")
    if not all(k in content['opinion_texts'] for k in ['positive', 'negative', 'balanced']):
        raise ValueError("opinion_texts must have 'positive', 'negative', 'balanced'")
    if not isinstance(content['discussion_questions'], list):
        raise ValueError("discussion_questions must be a list")
    return True

@retry(
    stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: print(f"Retry attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}")
)
def generate_content_with_deepseek(topic):
    print(f"[DeepSeek] Generating content for topic: {topic[:50]}...")
    prompt = f"""You are a Chinese language teaching assistant. Create learning materials about the topic: "{topic}"

Please generate a JSON response with the following structure:
{{
  "main_text": "A text in Simplified Chinese at HSK5 level about {topic}. Should be 250 characters long, natural and engaging.",
  "vocabulary": [
    {{"english": "English translation", "chinese": "Chinese word/phrase from the text", "pinyin": "pinyin with tone marks"}},
    // 10-15 items total - must be HSK5 level collocations or phrases are preferable, expressions taken directly from the main_text
  ],
  "opinion_texts": {{
    "positive": "A natural Chinese text (HSK5 level, 100-150 characters) giving a positive perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context. Use some of the vocabulary taken from the first text.",
    "negative": "A natural Chinese text (HSK5 level, 100-150 characters) giving a critical/negative perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context. Use some of the vocabulary taken from the first text.",
    "balanced": "A natural Chinese text (HSK5 level, 100-150 characters) giving a balanced perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context. Use some of the vocabulary taken from the first text."
  }},
  "discussion_questions": [
    "Question 1 in Chinese (HSK5 level) - should prompt discussion, not just comprehension",
    "Question 2 in Chinese (HSK5 level) - should prompt discussion, not just comprehension",
    "Question 3 in Chinese (HSK5 level) - should prompt discussion, not just comprehension",
    "Question 4 in Chinese (HSK5 level) - should prompt discussion, not just comprehension"
  ]
}}

Important requirements:
1. All vocabulary items MUST come from the main_text
2. Vocabulary should be HSK5 level collocations and phrases (not single words)
3. Opinion texts should use some of the vocabulary taken from the first text but should sound natural - vocabulary can be adjusted to fit context
4. Discussion questions should encourage personal opinions and deeper thinking
5. Return ONLY valid JSON, no additional text"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Chinese language teaching expert who creates engaging, natural content at HSK5 level. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            timeout=45.0
        )
        content_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            content_text = json_match.group()
        content = json.loads(content_text)
        validate_deepseek_response(content)
        return content
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing error: {str(e)}")
        print(f"[ERROR] Raw content: {content_text[:200]}...")
        raise
    except ValueError as e:
        print(f"[ERROR] Validation error: {str(e)}")
        raise
    except Exception as e:
        print(f"[ERROR] DeepSeek API Error: {type(e).__name__}: {str(e)}")
        raise

async def create_vocabulary_file_with_tts(vocabulary, topic, progress_callback=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_truncated = topic[:50] if len(topic) > 50 else topic
    safe_topic_name = safe_filename(topic_truncated)
    filename = f"{safe_topic_name}_{timestamp}_vocabulary.txt"
    content = ""
    audio_files = {}
    total_items = len(vocabulary)
    tts_tasks = []
    for item in vocabulary:
        tts_tasks.append(generate_tts_async(item['chinese'], voice_name=config.ANKI_VOICE, speaking_rate=0.8))
    audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    for idx, (item, audio_data) in enumerate(zip(vocabulary, audio_results)):
        chinese_text = item['chinese']
        if progress_callback:
            await progress_callback(idx + 1, total_items)
        if isinstance(audio_data, Exception) or not audio_data:  # FIXED: Complete variable name
            error_msg = str(audio_data) if isinstance(audio_data, Exception) else "No audio data"
            print(f"[VOCAB TTS] ❌ Failed for '{chinese_text}': {error_msg}")
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
        else:
            hash_object = hashlib.md5(chinese_text.encode())
            audio_filename = f"tts_{hash_object.hexdigest()}.mp3"
            audio_filename = safe_filename(audio_filename)
            audio_files[audio_filename] = audio_data
            anki_tag = f"[sound:{audio_filename}]"
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\t{anki_tag}\n"
    return filename, content, audio_files

def create_html_document(topic, content, timestamp):
    topic_truncated = topic[:50] if len(topic) > 50 else topic
    safe_topic = safe_filename(topic_truncated)
    html_filename = f"{safe_topic}_{timestamp}_materials.html"
    vocab_rows = ""
    for i, item in enumerate(content['vocabulary'], 1):
        vocab_rows += f"""
        <tr>
            <td>{i}</td>
            <td class="chinese">{item['chinese']}</td>
            <td class="pinyin">{item['pinyin']}</td>
            <td>{item['english']}</td>
        </tr>
        """
    questions_html = ""
    for i, question in enumerate(content['discussion_questions'], 1):
        questions_html += f"""
        <div class="question">
            <span class="question-number">{i}</span>
            <span class="question-text">{question}</span>
        </div>
        """
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chinese Learning Materials: {topic}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            line-height: 1.8;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        .header .subtitle {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 50px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .main-text {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 30px;
            border-radius: 15px;
            font-size: 1.3em;
            line-height: 2;
            color: #2c3e50;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }}
        .chinese {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }}
        .pinyin {{
            color: #7f8c8d;
            font-style: italic;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background-color: #f8f9fa;
            color: #667eea;
        }}
        .question {{
            margin-bottom: 15px;
            padding: 12px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .question-number {{
            font-weight: bold;
            color: #667eea;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{topic}</h1>
            <div class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
        </div>
        <div class="content">
            <div class="section">
                <div class="section-title">📖 Main Text</div>
                <div class="main-text">{content['main_text']}</div>
            </div>
            <div class="section">
                <div class="section-title">📚 Vocabulary</div>
                <table>
                    <thead>
                        <tr>
                            <th>#</th>
                            <th>Chinese</th>
                            <th>Pinyin</th>
                            <th>English</th>
                        </tr>
                    </thead>
                    <tbody>
                        {vocab_rows}
                    </tbody>
                </table>
            </div>
            <div class="section">
                <div class="section-title">💬 Opinion Texts</div>
                <div><strong>Positive:</strong> {content['opinion_texts']['positive']}</div>
                <div style="margin-top:10px;"><strong>Negative:</strong> {content['opinion_texts']['negative']}</div>
                <div style="margin-top:10px;"><strong>Balanced:</strong> {content['opinion_texts']['balanced']}</div>
            </div>
            <div class="section">
                <div class="section-title">❓ Discussion Questions</div>
                {questions_html}
            </div>
        </div>
    </div>
</body>
</html>"""
    return html_filename, html_content

# Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🇨🇳 欢迎！发送任何话题，我会为你生成中文学习材料，包括：\n\n"
        "• 中文课文 (HSK5水平)\n"
        "• 词汇表 (带拼音和音频)\n"
        "• 不同观点的短文\n"
        "• 讨论问题\n\n"
        "请输入一个话题："
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if not rate_limiter.is_allowed(user_id):
        reset_time = rate_limiter.get_reset_time(user_id)
        await update.message.reply_text(
            f"⏰ 请求过于频繁。请等待 {reset_time} 秒后再试。"
        )
        return

    try:
        topic = validate_topic(update.message.text)
        await update.message.reply_text(f"🔄 正在为『{topic}』生成学习材料...")
        
        # Generate content
        content = await asyncio.get_event_loop().run_in_executor(
            None, generate_content_with_deepseek, topic
        )
        
        # Create files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create vocabulary file with TTS
        vocab_filename, vocab_content, audio_files = await create_vocabulary_file_with_tts(
            content['vocabulary'], topic
        )
        
        # Create HTML document
        html_filename, html_content = create_html_document(topic, content, timestamp)
        
        # Create zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add vocabulary file
            zip_file.writestr(vocab_filename, vocab_content)
            
            # Add audio files
            for audio_filename, audio_data in audio_files.items():
                zip_file.writestr(audio_filename, audio_data)
            
            # Add HTML file
            zip_file.writestr(html_filename, html_content)
            
            # Add main text as separate file
            main_text_filename = f"{safe_filename(topic)}_{timestamp}_main_text.txt"
            zip_file.writestr(main_text_filename, content['main_text'])
        
        zip_buffer.seek(0)
        
        # Send zip file
        await update.message.reply_document(
            document=zip_buffer,
            filename=f"{safe_filename(topic)}_{timestamp}_materials.zip",
            caption=f"📚 学习材料: {topic}\n\n包含：课文、词汇表、音频、HTML文档"
        )
        
    except ValueError as e:
        await update.message.reply_text(f"❌ 错误: {str(e)}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        await update.message.reply_text("❌ 生成材料时出现错误，请稍后重试。")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"[ERROR] {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("❌ 发生意外错误，请稍后重试。")

def main():
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_error_handler(error_handler)
    
    print("🤖 Bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
