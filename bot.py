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
    TTS_TIMEOUT = 30
    API_RETRY_ATTEMPTS = 3
    RATE_LIMIT_REQUESTS = 5
    RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB for Telegram
    
    # Chirp3 voices for main text and opinion texts (random selection)
    # Updated list - removed potentially problematic voices
    CHIRP3_VOICES = [
        "cmn-CN-Chirp3-HD-Aoede",
        "cmn-CN-Chirp3-HD-Leda",
        "cmn-CN-Chirp3-HD-Puck"
    ]
    
    # Backup voices if primary fails
    CHIRP3_BACKUP_VOICES = [
        "cmn-CN-Chirp3-HD-Leda",  # Most reliable
        "cmn-CN-Chirp3-HD-Aoede"
    ]
    
    # Fixed voice for Anki vocabulary cards
    ANKI_VOICE = "cmn-CN-Chirp3-HD-Leda"
    
    # Fallback to older generation if Chirp3 fails completely
    FALLBACK_VOICES = [
        "cmn-CN-Wavenet-A",
        "cmn-CN-Wavenet-B"
    ]

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
        
        # Remove old requests outside the time window
        user_requests[:] = [req_time for req_time in user_requests 
                          if now - req_time < self.window]
        
        if len(user_requests) >= self.max_requests:
            return False
        
        user_requests.append(now)
        return True
    
    def get_reset_time(self, user_id):
        """Get time until rate limit resets"""
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
    """Initialize Google TTS client with credentials from environment variable"""
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
    """Validate and sanitize topic input"""
    # Remove excessive whitespace
    topic = re.sub(r'\s+', ' ', topic.strip())
    
    # Check for harmful patterns (command injection, path traversal)
    if re.search(r'[<>"|&;`$()]', topic):
        raise ValueError("Topic contains invalid characters")
    
    # Basic content moderation (Chinese and English)
    inappropriate_patterns = [
        r'\b(porn|sex|violence|hate|kill|death)\b',
        r'\b(暴力|色情|仇恨|歧视|杀|死)\b',
    ]
    
    for pattern in inappropriate_patterns:
        if re.search(pattern, topic, re.IGNORECASE):
            raise ValueError("Topic contains inappropriate content")
    
    # Enforce length limit
    if len(topic) > config.MAX_TOPIC_LENGTH:
        topic = topic[:config.MAX_TOPIC_LENGTH]
    
    if not topic:
        raise ValueError("Topic cannot be empty")
    
    return topic

def split_text_into_sentences(text, max_length=150):
    """Split text into smaller sentences for Chirp3"""
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

def generate_tts_chirp3_sync(text, voice_name=None, speaking_rate=1.0, use_chirp=True):
    """
    Generate Chinese TTS audio using Google Cloud Chirp3 (sync version) with fallback
    
    Args:
        text: Chinese text to convert to speech
        voice_name: Specific Chirp3 voice to use. If None, randomly selects from config.CHIRP3_VOICES
        speaking_rate: Speed of speech (0.25 to 2.0). Default 1.0. Lower = slower.
        use_chirp: If True, try Chirp3 voices. If False or fails, use fallback voices.
    
    Returns:
        tuple: (audio_content, voice_used, success)
    """
    try:
        client = get_google_tts_client()
        
        voices_to_try = []
        
        if use_chirp:
            # Select voice: use provided voice_name or randomly select one
            if voice_name is None:
                primary_voice = random.choice(config.CHIRP3_VOICES)
            else:
                primary_voice = voice_name
            
            # Build list of voices to try
            voices_to_try.append(primary_voice)
            
            # Add backup voices (but don't repeat the primary)
            for backup in config.CHIRP3_BACKUP_VOICES:
                if backup != primary_voice and backup not in voices_to_try:
                    voices_to_try.append(backup)
        
        # Add fallback voices
        voices_to_try.extend(config.FALLBACK_VOICES)
        
        # Clamp speaking_rate to valid range [0.25, 2.0]
        speaking_rate = max(0.25, min(2.0, speaking_rate))
        
        last_error = None
        
        # Try each voice in order
        for current_voice in voices_to_try:
            try:
                print(f"[TTS] Attempting voice: {current_voice} for text: '{text[:30]}...' at speed {speaking_rate}")
                
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
                    raise ValueError(f"TTS returned empty audio content")
                
                print(f"[TTS] ✅ SUCCESS with voice: {current_voice} ({len(response.audio_content)} bytes)")
                return response.audio_content, current_voice, True
            
            except Exception as e:
                last_error = e
                print(f"[TTS] ❌ FAILED with voice {current_voice}: {type(e).__name__}: {str(e)}")
                # Continue to next voice
                continue
        
        # If we got here, all voices failed
        print(f"[TTS ERROR] All voices failed for text: '{text[:50]}...'")
        if last_error:
            raise last_error
        else:
            raise Exception("All TTS voices failed")
    
    except Exception as e:
        print(f"[TTS ERROR FINAL] Text: '{text[:50]}...', Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, False

async def generate_tts_async(text, voice_name=None, speaking_rate=1.0):
    """
    Run TTS generation in thread pool
    
    Args:
        text: Chinese text to convert to speech
        voice_name: Specific Chirp3 voice to use. If None, randomly selects
        speaking_rate: Speed of speech (0.25 to 2.0). Default 1.0.
    
    Returns:
        tuple: (audio_content, voice_used, success)
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, generate_tts_chirp3_sync, text, voice_name, speaking_rate)

def safe_filename(filename):
    """Sanitize filename to prevent path traversal (ZIP slip vulnerability)"""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    # Get just the basename to strip any path components
    filename = os.path.basename(filename)
    # Ensure reasonable length
    filename = filename[:100]
    return filename.strip('_')

def validate_deepseek_response(content):
    """Validate DeepSeek JSON response structure"""
    required_keys = ["main_text", "vocabulary", "opinion_texts", "discussion_questions"]
    
    # Check all required keys exist
    if not all(k in content for k in required_keys):
        missing = [k for k in required_keys if k not in content]
        raise ValueError(f"Missing required keys in DeepSeek response: {missing}")
    
    # Validate vocabulary is a list
    if not isinstance(content['vocabulary'], list):
        raise ValueError("vocabulary must be a list")
    
    # Limit vocabulary items
    if len(content['vocabulary']) > config.MAX_VOCAB_ITEMS:
        content['vocabulary'] = content['vocabulary'][:config.MAX_VOCAB_ITEMS]
    
    # Validate each vocabulary item has required fields
    for item in content['vocabulary']:
        if not all(k in item for k in ['english', 'chinese', 'pinyin']):
            raise ValueError("Each vocabulary item must have 'english', 'chinese', 'pinyin'")
    
    # Validate opinion_texts has all three views
    if not all(k in content['opinion_texts'] for k in ['positive', 'negative', 'balanced']):
        raise ValueError("opinion_texts must have 'positive', 'negative', 'balanced'")
    
    # Validate discussion_questions is a list
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
    """Generate all content using DeepSeek API with retry logic"""
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
        print(f"[DeepSeek] Sending request to API...")
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Chinese language teaching expert who creates engaging, natural content at HSK5 level. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            timeout=45.0
        )
        
        print(f"[DeepSeek] Received response, parsing...")
        content_text = response.choices[0].message.content
        
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            content_text = json_match.group()
        
        # Parse JSON
        content = json.loads(content_text)
        
        print(f"[DeepSeek] JSON parsed successfully")
        
        # Validate structure
        validate_deepseek_response(content)
        
        print(f"[DeepSeek] Validation passed, returning content")
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
    """
    Create tab-delimited vocabulary file with TTS audio tags and return audio files
    Uses fixed Leda voice for all Anki vocabulary cards
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Truncate topic properly before sanitizing
    topic_truncated = topic[:50] if len(topic) > 50 else topic
    safe_topic_name = safe_filename(topic_truncated)
    filename = f"{safe_topic_name}_{timestamp}_vocabulary.txt"
    
    content = ""
    audio_files = {}
    voice_info = []  # Track which voice was used for each item
    
    total_items = len(vocabulary)
    
    # Generate TTS for all vocabulary items concurrently using Leda voice at 80% speed
    tts_tasks = []
    for item in vocabulary:
        # Use the fixed Anki voice (Leda) at 0.8 speed for vocabulary cards
        tts_tasks.append(generate_tts_async(item['chinese'], voice_name=config.ANKI_VOICE, speaking_rate=0.8))
    
    # Await all TTS generations
    audio_results = await asyncio.gather(*tts_tasks, return_exceptions=True)
    
    for idx, (item, result) in enumerate(zip(vocabulary, audio_results)):
        chinese_text = item['chinese']
        
        if progress_callback:
            await progress_callback(idx + 1, total_items)
        
        # Handle result tuple or exception
        if isinstance(result, Exception):
            print(f"TTS failed for '{chinese_text}': {result}")
            voice_info.append(f"❌ {chinese_text}: FAILED - {str(result)[:50]}")
            # Add row without audio
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
        elif isinstance(result, tuple):
            audio_data, voice_used, success = result
            if success and audio_data:
                # Create filename using MD5 hash
                hash_object = hashlib.md5(chinese_text.encode())
                audio_filename = f"tts_{hash_object.hexdigest()}.mp3"
                
                # Sanitize filename
                audio_filename = safe_filename(audio_filename)
                
                # Store audio data
                audio_files[audio_filename] = audio_data
                
                # Create Anki sound tag
                anki_tag = f"[sound:{audio_filename}]"
                
                # Add row with 4 columns: english, chinese, pinyin, audio_tag
                content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\t{anki_tag}\n"
                voice_info.append(f"✅ {chinese_text}: {voice_used}")
            else:
                voice_info.append(f"❌ {chinese_text}: FAILED - no audio data")
                content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
        else:
            voice_info.append(f"❌ {chinese_text}: FAILED - unexpected result")
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
    
    return filename, content, audio_files, voice_info

def create_html_document(topic, content, timestamp, voice_info=None):
    """Create a beautiful HTML document with all learning materials"""
    # Truncate topic properly before sanitizing
    topic_truncated = topic[:50] if len(topic) > 50 else topic
    safe_topic = safe_filename(topic_truncated)
    html_filename = f"{safe_topic}_{timestamp}_materials.html"
    
    # Build vocabulary table HTML
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
    
    # Build discussion questions HTML
    questions_html = ""
    for i, question in enumerate(content['discussion_questions'], 1):
        questions_html += f"""
        <div class="question">
            <span class="question-number">{i}</span>
            <span class="question-text">{question}</span>
        </div>
        """
    
    # Add voice info section if provided
    voice_info_html = ""
    if voice_info:
        voice_info_html = """
        <div class="section">
            <h2 class="section-title">
                <span class="section-icon">🎙️</span>
                TTS Voice Information
            </h2>
            <div class="voice-info">
"""
        for info in voice_info:
            voice_info_html += f"                <div class='voice-item'>{info}</div>\n"
        voice_info_html += """
            </div>
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
        
        .section-icon {{
            font-size: 1.2em;
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
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-radius: 10px;
            overflow: hidden;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        tr:hover {{
            background-color: #f8f9fa;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        .opinion-box {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            border-left: 5px solid;
        }}
        
        .opinion-box.positive {{
            border-left-color: #27ae60;
        }}
        
        .opinion-box.negative {{
            border-left-color: #e74c3c;
        }}
        
        .opinion-box.balanced {{
            border-left-color: #3498db;
        }}
        
        .opinion-title {{
            font-size: 1.2em;
            font-weight: 600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .opinion-text {{
            font-size: 1.1em;
            line-height: 2;
            color: #2c3e50;
        }}
        
        .question {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
            display: flex;
            gap: 15px;
            align-items: start;
        }}
        
        .question-number {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            flex-shrink: 0;
        }}
        
        .question-text {{
            flex: 1;
            font-size: 1.1em;
            line-height: 1.6;
            color: #2c3e50;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 30px;
            text-align: center;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
        }}
        
        .voice-info {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            font-family: monospace;
            font-size: 0.9em;
        }}
        
        .voice-item {{
            padding: 5px 0;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .voice-item:last-child {{
            border-bottom: none;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            
            .container {{
                box-shadow: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Chinese Learning Materials</h1>
            <div class="subtitle">Topic: {topic}</div>
            <div class="subtitle">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
        </div>
        
        <div class="content">
            <!-- Main Text -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">📖</span>
                    Main Text
                </h2>
                <div class="main-text">
                    {content['main_text']}
                </div>
            </div>
            
            <!-- Vocabulary -->
            <div class="section">
                <h2 class="section-title">
                    <span class="section-icon">📝</span>
                    Key Vocabulary
                </h2>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 60px;">#</th>
                            <th style="width: 30%;">Chinese</th>
                            <th style="width: 30%;">Pinyin</th>
                            <th style="width: 40%;">English</th>
                        </tr>
                    </thead>
                    <tbody>
                        {vocab_rows}
