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

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

# Configuration
class Config:
    MAX_TOPIC_LENGTH = 100
    MAX_VOCAB_ITEMS = 15
    TTS_TIMEOUT = 30
    API_RETRY_ATTEMPTS = 3
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 3600
    MAX_FILE_SIZE = 50 * 1024 * 1024
    
    # Primary Chirp3 voices (reduced to most reliable)
    CHIRP3_VOICES = [
        "cmn-CN-Chirp3-HD-Aoede",
        "cmn-CN-Chirp3-HD-Leda",
        "cmn-CN-Chirp3-HD-Puck"
    ]
    
    # Backup voices
    CHIRP3_BACKUP_VOICES = [
        "cmn-CN-Chirp3-HD-Leda",
        "cmn-CN-Chirp3-HD-Aoede"
    ]
    
    ANKI_VOICE = "cmn-CN-Chirp3-HD-Leda"
    
    # Fallback to Wavenet if Chirp3 fails
    FALLBACK_VOICES = [
        "cmn-CN-Wavenet-A",
        "cmn-CN-Wavenet-B"
    ]

config = Config()

deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

class RateLimiter:
    def __init__(self, max_requests=5, window=3600):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
    
    def is_allowed(self, user_id):
        now = time.time()
        user_requests = self.requests[user_id]
        user_requests[:] = [req_time for req_time in user_requests 
                          if now - req_time < self.window]
        
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

def generate_tts_chirp3_sync(text, voice_name=None, speaking_rate=1.0):
    """Generate TTS with fallback voices and return (audio, voice_used, success)"""
    try:
        client = get_google_tts_client()
        
        voices_to_try = []
        
        if voice_name is None:
            primary_voice = random.choice(config.CHIRP3_VOICES)
        else:
            primary_voice = voice_name
        
        voices_to_try.append(primary_voice)
        
        for backup in config.CHIRP3_BACKUP_VOICES:
            if backup != primary_voice and backup not in voices_to_try:
                voices_to_try.append(backup)
        
        voices_to_try.extend(config.FALLBACK_VOICES)
        
        speaking_rate = max(0.25, min(2.0, speaking_rate))
        
        last_error = None
        
        for current_voice in voices_to_try:
            try:
                print(f"[TTS] Trying voice: {current_voice} for '{text[:30]}...' @ {speaking_rate}x")
                
                synthesis_input = texttospeech.SynthesisInput(text=text)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="cmn-CN",
                    name=current_voice
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=speaking_rate
                )
                
                response = client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                    timeout=config.TTS_TIMEOUT
                )
                
                if not response.audio_content:
                    raise ValueError("Empty audio content")
                
                print(f"[TTS] ✅ SUCCESS: {current_voice} ({len(response.audio_content)} bytes)")
                return response.audio_content, current_voice, True
            
            except Exception as e:
                last_error = e
                print(f"[TTS] ❌ FAILED: {current_voice} - {type(e).__name__}: {str(e)}")
                continue
        
        print(f"[TTS ERROR] All voices failed for '{text[:50]}...'")
        if last_error:
            raise last_error
        else:
            raise Exception("All TTS voices failed")
    
    except Exception as e:
        print(f"[TTS ERROR FINAL] '{text[:50]}...': {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, False

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
        raise ValueError(f"Missing required keys: {missing}")
    
    if not isinstance(content['vocabulary'], list):
        raise ValueError("vocabulary must be a list")
    
    if len(content['vocabulary']) > config.MAX_VOCAB_ITEMS:
        content['vocabulary'] = content['vocabulary'][:config.MAX_VOCAB_ITEMS]
    
    for item in content['vocabulary']:
        if not all(k in item for k in ['english', 'chinese', 'pinyin']):
            raise ValueError("Each vocab item needs 'english', 'chinese', 'pinyin'")
    
    if not all(k in content['opinion_texts'] for k in ['positive', 'negative', 'balanced']):
        raise ValueError("opinion_texts needs 'positive', 'negative', 'balanced'")
    
    if not isinstance(content['discussion_questions'], list):
        raise ValueError("discussion_questions must be a list")
    
    return True

@retry(
    stop=stop_after_attempt(config.API_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: print(f"Retry {retry_state.attempt_number}: {retry_state.outcome.exception()}")
)
def generate_content_with_deepseek(topic):
    print(f"[DeepSeek] Generating content for: {topic[:50]}...")
    
    prompt = f"""You are a Chinese language teaching assistant. Create learning materials about: "{topic}"

Generate JSON with this structure:
{{
  "main_text": "Simplified Chinese text at HSK5 level about {topic}. 250 characters, natural and engaging.",
  "vocabulary": [
    {{"english": "translation", "chinese": "word/phrase", "pinyin": "pinyin with tones"}},
    // 10-15 HSK5 collocations/phrases from main_text
  ],
  "opinion_texts": {{
    "positive": "HSK5 Chinese text (100-150 chars) with positive view, using 5-6 vocab items",
    "negative": "HSK5 Chinese text (100-150 chars) with critical view, using 5-6 vocab items",
    "balanced": "HSK5 Chinese text (100-150 chars) with balanced view, using 5-6 vocab items"
  }},
  "discussion_questions": [
    "Question 1 in Chinese (HSK5) - prompts discussion",
    "Question 2 in Chinese (HSK5) - prompts discussion",
    "Question 3 in Chinese (HSK5) - prompts discussion",
    "Question 4 in Chinese (HSK5) - prompts discussion"
  ]
}}

Requirements:
1. All vocab from main_text
2. HSK5 level collocations/phrases
3. Opinion texts use vocab naturally
4. Questions encourage deep thinking
5. ONLY valid JSON"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Chinese teaching expert. HSK5 level. Valid JSON only."},
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
        
        print(f"[DeepSeek] ✅ Content generated successfully")
        return content
    
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse: {str(e)}")
        print(f"[ERROR] Raw: {content_text[:200]}...")
        raise
    except Exception as e:
        print(f"[ERROR] DeepSeek: {type(e).__name__}: {str(e)}")
        raise

async def create_vocabulary_file_with_tts(vocabulary, topic, progress_callback=None):
    """Create vocab file with TTS, return (filename, content, audio_files, voice_info)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_truncated = topic[:50] if len(topic) > 50 else topic
    safe_topic_name = safe_filename(topic_truncated)
    filename = f"{safe_topic_name}_{timestamp}_vocabulary.txt"
    
    content = ""
    audio_files = {}
    voice_info = []
    
    total_items = len(vocabulary)
    
    tts_tasks = []
    for item in vocabulary:
        tts_tasks.append(generate_tts_async(item['chinese'], voice_name=config.ANKI_VOICE, speaking_rate=0.8))
    
    audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    
    for idx, (item, result) in enumerate(zip(vocabulary, audio_results)):
        chinese_text = item['chinese']
        
        if progress_callback:
            await progress_callback(idx + 1, total_items)
        
        if isinstance(result, Exception):
            print(f"TTS failed for '{chinese_text}': {result}")
            voice_info.append(f"❌ {chinese_text}: FAILED - {str(result)[:50]}")
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
        elif isinstance(result, tuple):
            audio_data, voice_used, success = result
            if success and audio_data:
                hash_object = hashlib.md5(chinese_text.encode())
                audio_filename = f"tts_{hash_object.hexdigest()}.mp3"
                audio_filename = safe_filename(audio_filename)
                
                audio_files[audio_filename] = audio_data
                anki_tag = f"[sound:{audio_filename}]"
                
                content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\t{anki_tag}\n"
                voice_info.append(f"✅ {chinese_text}: {voice_used}")
            else:
                voice_info.append(f"❌ {chinese_text}: FAILED - no audio")
                content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
        else:
            voice_info.append(f"❌ {chinese_text}: FAILED - unexpected result")
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
    
    return filename, content, audio_files, voice_info

# Note: The rest of your code (HTML generation, handlers, main, etc.) continues here
# I'll provide a simplified version to fit within limits

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = """👋 Welcome to Chinese Learning Bot!

Send me any topic and I'll create:
📖 Main text (HSK5 level)
📝 Vocabulary with audio
💭 Multiple perspectives
❓ Discussion questions
📦 Anki-ready package

Just send a topic to begin!"""
    
    await update.message.reply_text(welcome_msg)

def create_html_document(topic, content, timestamp):
    """Create HTML document"""
    topic_truncated = topic[:50] if len(topic) > 50 else topic
    safe_topic = safe_filename(topic_truncated)
    html_filename = f"{safe_topic}_{timestamp}_materials.html"
    
    vocab_rows = ""
    for i, item in enumerate(content['vocabulary'], 1):
        vocab_rows += f"<tr><td>{i}</td><td class='chinese'>{item['chinese']}</td><td class='pinyin'>{item['pinyin']}</td><td>{item['english']}</td></tr>\n"
    
    questions_html = ""
    for i, question in enumerate(content['discussion_questions'], 1):
        questions_html += f"<div class='question'><span class='question-number'>{i}</span><span class='question-text'>{question}</span></div>\n"
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chinese Learning: {topic}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Hiragino Sans GB','Microsoft YaHei',sans-serif;line-height:1.8;color:#333;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px;min-height:100vh}}
.container{{max-width:900px;margin:0 auto;background:white;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3);overflow:hidden}}
.header{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:40px;text-align:center}}
.header h1{{font-size:2em;margin-bottom:10px;text-shadow:2px 2px 4px rgba(0,0,0,0.2)}}
.content{{padding:40px}}
.section{{margin-bottom:50px}}
.section-title{{font-size:1.8em;color:#667eea;margin-bottom:20px;padding-bottom:10px;border-bottom:3px solid #667eea}}
.main-text{{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);padding:30px;border-radius:15px;font-size:1.3em;line-height:2;color:#2c3e50;box-shadow:0 5px 15px rgba(0,0,0,0.1)}}
.chinese{{font-size:1.2em;font-weight:600;color:#2c3e50}}
.pinyin{{color:#7f8c8d;font-style:italic}}
table{{width:100%;border-collapse:collapse;margin-top:20px;box-shadow:0 5px 15px rgba(0,0,0,0.1);border-radius:10px;overflow:hidden}}
th{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:15px;text-align:left;font-weight:600}}
td{{padding:15px;border-bottom:1px solid #ecf0f1}}
tr:hover{{background-color:#f8f9fa}}
tr:last-child td{{border-bottom:none}}
.opinion-box{{background:white;border-radius:15px;padding:25px;margin-bottom:25px;box-shadow:0 5px 15px rgba(0,0,0,0.1);border-left:5px solid}}
.opinion-box.positive{{border-left-color:#27ae60}}
.opinion-box.negative{{border-left-color:#e74c3c}}
.opinion-box.balanced{{border-left-color:#3498db}}
.opinion-title{{font-size:1.2em;font-weight:600;margin-bottom:15px}}
.opinion-text{{font-size:1.1em;line-height:2;color:#2c3e50}}
.question{{background:white;padding:20px;border-radius:10px;margin-bottom:15px;box-shadow:0 3px 10px rgba(0,0,0,0.1);display:flex;gap:15px}}
.question-number{{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;width:35px;height:35px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:600;flex-shrink:0}}
.question-text{{flex:1;font-size:1.1em;line-height:1.6;color:#2c3e50}}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>{topic}</h1></div>
<div class="content">
<div class="section"><h2 class="section-title">📖 Main Text</h2><div class="main-text">{content['main_text']}</div></div>
<div class="section"><h2 class="section-title">📝 Vocabulary</h2><table><thead><tr><th>#</th><th>Chinese</th><th>Pinyin</th><th>English</th></tr></thead><tbody>{vocab_rows}</tbody></table></div>
<div class="section"><h2 class="section-title">💭 Perspectives</h2>
<div class="opinion-box positive"><div class="opinion-title">✅ Positive View</div><div class="opinion-text">{content['opinion_texts']['positive']}</div></div>
<div class="opinion-box negative"><div class="opinion-title">⚠️ Critical View</div><div class="opinion-text">{content['opinion_texts']['negative']}</div></div>
<div class="opinion-box balanced"><div class="opinion-title">⚖️ Balanced View</div><div class="opinion-text">{content['opinion_texts']['balanced']}</div></div>
</div>
<div class="section"><h2 class="section-title">❓ Discussion Questions</h2>{questions_html}</div>
</div></div></body></html>"""
    
    return html_filename, html_content

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle topic messages"""
    user_id = update.effective_user.id
    
    if not rate_limiter.is_allowed(user_id):
        reset_time = rate_limiter.get_reset_time(user_id)
        await update.message.reply_text(
            f"⏱️ Rate limit reached. Try again in {reset_time // 60} minutes."
        )
        return
    
    try:
        topic = validate_topic(update.message.text)
        
        status_msg = await update.message.reply_text(
            "🔄 Generating materials..."
        )
        
        # Generate content
        content = generate_content_with_deepseek(topic)
        
        await status_msg.edit_text("✅ Content generated\n🎙️ Creating audio files...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic_name = safe_filename(topic[:50])
        
        # 1. Generate main text TTS (random Chirp3 voice)
        await status_msg.edit_text("🎙️ Creating main text audio...")
        main_audio_data, main_voice_used, main_success = await generate_tts_async(
            content['main_text'],
            voice_name=None,  # Random Chirp3
            speaking_rate=0.9
        )
        
        # 2. Generate opinion TTS (random Chirp3 voices)
        await status_msg.edit_text("🎙️ Creating opinion audio files...")
        opinion_audio_data = {}
        opinion_voice_info = []
        
        for opinion_type in ['positive', 'negative', 'balanced']:
            opinion_text = content['opinion_texts'][opinion_type]
            audio_data, voice_used, success = await generate_tts_async(
                opinion_text, 
                voice_name=None,  # Random Chirp3
                speaking_rate=0.9
            )
            
            if success and audio_data:
                opinion_audio_data[opinion_type] = audio_data
                opinion_voice_info.append(f"✅ {opinion_type}: {voice_used}")
            else:
                opinion_voice_info.append(f"❌ {opinion_type}: FAILED")
        
        # 3. Create vocabulary with TTS (Leda voice for Anki)
        await status_msg.edit_text("🎙️ Creating vocabulary audio...")
        vocab_filename, vocab_content, vocab_audio_files, vocab_voice_info = \
            await create_vocabulary_file_with_tts(content['vocabulary'], topic)
        
        # 4. Create HTML
        await status_msg.edit_text("📝 Creating HTML document...")
        html_filename, html_content = create_html_document(topic, content, timestamp)
        
        # Send voice info report
        voice_report = "🎙️ **TTS Voice Report**\n\n"
        if main_success:
            voice_report += f"**Main Text:** ✅ {main_voice_used}\n\n"
        else:
            voice_report += "**Main Text:** ❌ FAILED\n\n"
        
        voice_report += "**Opinion Texts:**\n" + "\n".join(opinion_voice_info)
        voice_report += f"\n\n**Vocabulary:** {len(vocab_audio_files)} audio files created"
        
        await update.message.reply_text(voice_report)
        
        # Now send files in order:
        await status_msg.edit_text("📤 Sending files...")
        
        # 1. HTML file
        html_buffer = BytesIO(html_content.encode('utf-8'))
        html_buffer.seek(0)
        await update.message.reply_document(
            document=html_buffer,
            filename=html_filename
        )
        
        # 2. Main text audio
        if main_success and main_audio_data:
            main_audio_filename = f"{safe_topic_name}_{timestamp}_main.mp3"
            main_audio_buffer = BytesIO(main_audio_data)
            main_audio_buffer.seek(0)
            await update.message.reply_audio(
                audio=main_audio_buffer,
                filename=main_audio_filename
            )
        
        # 3. Opinion audio files (positive, negative, balanced)
        for opinion_type in ['positive', 'negative', 'balanced']:
            if opinion_type in opinion_audio_data:
                audio_filename = f"{safe_topic_name}_{timestamp}_{opinion_type}.mp3"
                audio_buffer = BytesIO(opinion_audio_data[opinion_type])
                audio_buffer.seek(0)
                await update.message.reply_audio(
                    audio=audio_buffer,
                    filename=audio_filename
                )
        
        # 4. Anki vocabulary text file
        vocab_buffer = BytesIO(vocab_content.encode('utf-8'))
        vocab_buffer.seek(0)
        await update.message.reply_document(
            document=vocab_buffer,
            filename=vocab_filename
        )
        
        # 5. ZIP file with ONLY Anki TTS audio files
        await status_msg.edit_text("📦 Creating Anki audio package...")
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Only add vocabulary audio files (for Anki)
            for filename, audio_data in vocab_audio_files.items():
                zip_file.writestr(filename, audio_data)
        
        zip_buffer.seek(0)
        zip_filename = f"{safe_topic_name}_{timestamp}_anki_audio.zip"
        
        await update.message.reply_document(
            document=zip_buffer,
            filename=zip_filename
        )
        
        await status_msg.delete()
        
    except ValueError as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        await update.message.reply_text(
            "❌ An error occurred. Please try again or contact support."
        )

def main():
    """Start the bot"""
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 Bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
