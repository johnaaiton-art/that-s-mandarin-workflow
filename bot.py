import os
import json
import hashlib
import re
import zipfile
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
from google.cloud import texttospeech
from google.oauth2 import service_account
import asyncio
from io import BytesIO

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON")

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
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
        # Fallback to default credentials if available
        return texttospeech.TextToSpeechClient()

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

def generate_tts_chirp(text):
    """Generate Chinese TTS audio using Google Cloud Chirp3"""
    try:
        client = get_google_tts_client()
        
        sentences = split_text_into_sentences(text, max_length=150)
        
        all_audio = b""
        for sentence in sentences:
            synthesis_input = texttospeech.SynthesisInput(text=sentence)
            voice = texttospeech.VoiceSelectionParams(
                language_code="cmn-CN",
                name="cmn-CN-Chirp3-HD-Aoede",
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
            )
            
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3
            )
            
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            all_audio += response.audio_content
        
        return all_audio
    
    except Exception as e:
        print(f"Chirp3 TTS Error: {str(e)}")
        # Try fallback to Wavenet
        return generate_tts_wavenet(text)

def generate_tts_wavenet(text):
    """Fallback TTS using Wavenet"""
    try:
        client = get_google_tts_client()
        
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="cmn-CN",
            name="cmn-CN-Wavenet-A",
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=0.8,
        )
        
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )
        
        return response.audio_content
    
    except Exception as e:
        print(f"Wavenet TTS Error: {str(e)}")
        return None

def generate_vocabulary_tts(chinese_text):
    """Generate TTS for vocabulary items"""
    return generate_tts_wavenet(chinese_text)

def safe_filename(filename):
    """Sanitize filename to prevent path traversal (ZIP slip vulnerability)"""
    # Remove path separators and dangerous characters
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace('..', '').replace('/', '').replace('\\', '')
    # Get just the basename to strip any path components
    filename = os.path.basename(filename)
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

def generate_content_with_deepseek(topic):
    """Generate all content using DeepSeek API"""
    prompt = f"""You are a Chinese language teaching assistant. Create learning materials about the topic: "{topic}"

Please generate a JSON response with the following structure:
{{
  "main_text": "A text in Simplified Chinese at HSK4 level about {topic}. Should be 150-200 characters long, natural and engaging.",
  "vocabulary": [
    {{"english": "English translation", "chinese": "Chinese word/phrase from the text", "pinyin": "pinyin with tone marks"}},
    // 10-15 items total - must be HSK4 or HSK5 level words, phrases, or expressions taken directly from the main_text
  ],
  "opinion_texts": {{
    "positive": "A natural Chinese text (HSK4 level, 100-150 characters) giving a positive perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context.",
    "negative": "A natural Chinese text (HSK4 level, 100-150 characters) giving a critical/negative perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context.",
    "balanced": "A natural Chinese text (HSK4 level, 100-150 characters) giving a balanced perspective on the main topic. Must naturally incorporate at least 5-6 vocabulary items from the list, but adjust them to fit naturally in context."
  }},
  "discussion_questions": [
    "Question 1 in Chinese (HSK4 level) - should prompt discussion, not just comprehension",
    "Question 2 in Chinese (HSK4 level) - should prompt discussion, not just comprehension",
    "Question 3 in Chinese (HSK4 level) - should prompt discussion, not just comprehension",
    "Question 4 in Chinese (HSK4 level) - should prompt discussion, not just comprehension"
  ]
}}

Important requirements:
1. All vocabulary items MUST come from the main_text
2. Vocabulary should be HSK4-5 level (not easier levels)
3. Opinion texts should sound natural - vocabulary can be adjusted to fit context
4. Discussion questions should encourage personal opinions and deeper thinking
5. Return ONLY valid JSON, no additional text"""

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a Chinese language teaching expert who creates engaging, natural content at HSK4 level. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        content_text = response.choices[0].message.content
        
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
        if json_match:
            content_text = json_match.group()
        
        # Parse JSON
        content = json.loads(content_text)
        
        # Validate structure
        validate_deepseek_response(content)
        
        return content
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return None
    except ValueError as e:
        print(f"Validation error: {str(e)}")
        return None
    except Exception as e:
        print(f"DeepSeek API Error: {str(e)}")
        return None

def create_vocabulary_file_with_tts(vocabulary, topic):
    """Create tab-delimited vocabulary file with TTS audio tags and return audio files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic_name = safe_filename(topic)
    filename = f"{safe_topic_name}_{timestamp}_vocabulary.txt"
    
    content = ""
    audio_files = {}  # Dictionary to store audio data: {filename: audio_bytes}
    
    for item in vocabulary:
        chinese_text = item['chinese']
        
        # Generate TTS audio
        audio_data = generate_vocabulary_tts(chinese_text)
        
        if audio_data:
            # Create filename using MD5 hash (same as original code)
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
        else:
            # If TTS fails, just add 3 columns without audio
            content += f"{item['english']}\t{item['chinese']}\t{item['pinyin']}\n"
    
    return filename, content, audio_files

def create_zip_package(vocab_filename, vocab_content, audio_files, topic, timestamp):
    """Create a ZIP file containing vocabulary txt and all MP3 files"""
    safe_topic_name = safe_filename(topic)
    zip_filename = f"{safe_topic_name}_{timestamp}_anki_package.zip"
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Sanitize vocabulary filename
        safe_vocab_filename = safe_filename(vocab_filename)
        
        # Add vocabulary text file
        zip_file.writestr(safe_vocab_filename, vocab_content.encode('utf-8'))
        
        # Add all audio files with sanitized names
        for audio_filename, audio_data in audio_files.items():
            safe_audio_filename = safe_filename(audio_filename)
            zip_file.writestr(safe_audio_filename, audio_data)
    
    zip_buffer.seek(0)
    return zip_filename, zip_buffer

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "欢迎! Welcome to the Chinese Learning Bot!\n\n"
        "Simply send me a topic (e.g., '大城市生活的压力', 'environmental protection', etc.) "
        "and I'll create:\n"
        "📄 A reading text\n"
        "📝 Vocabulary list with TTS audio\n"
        "🎤 3 opinion texts with audio\n"
        "💬 Discussion questions\n"
        "📦 ZIP package for Anki import\n\n"
        "Just type your topic to begin!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "How to use:\n"
        "1. Send me any topic in English or Chinese (max 100 characters)\n"
        "2. Wait while I generate materials (takes about 30-60 seconds)\n"
        "3. You'll receive:\n"
        "   - Vocabulary file with TTS tags (.txt)\n"
        "   - 3 audio files with different perspectives\n"
        "   - 3 opinion texts\n"
        "   - Discussion questions\n"
        "   - ZIP package with vocabulary + MP3s for Anki\n\n"
        "📦 For Anki import:\n"
        "   - Download the ZIP file\n"
        "   - Extract MP3 files to your Anki collection.media folder\n"
        "   - Import the .txt file into Anki\n"
        "   - See Anki documentation for your platform's media folder location\n\n"
        "Example topics:\n"
        "- 社交媒体的影响\n"
        "- work-life balance\n"
        "- 环境保护\n"
        "- modern technology"
    )

async def handle_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle topic message and generate all materials"""
    topic = update.message.text.strip()
    
    # Input validation
    if len(topic) > 100:
        await update.message.reply_text("❌ Topic too long! Please keep it under 100 characters.")
        return
    
    if not topic:
        await update.message.reply_text("❌ Please send a topic to generate materials.")
        return
    
    # Send typing action
    await update.message.chat.send_action(action="typing")
    
    await update.message.reply_text(f"📚 正在创建关于 '{topic}' 的学习材料...\n"
                                   f"Creating materials about '{topic}'...\n"
                                   f"This will take about 30-60 seconds...")
    
    # Send periodic typing updates
    async def send_typing_updates():
        for _ in range(12):  # 60 seconds / 5 seconds each
            await asyncio.sleep(5)
            try:
                await update.message.chat.send_action(action="typing")
            except:
                break
    
    typing_task = asyncio.create_task(send_typing_updates())
    
    try:
        # Generate content with DeepSeek
        await update.message.reply_text("⏳ Step 1/4: Generating content with AI...")
        content = generate_content_with_deepseek(topic)
        
        if not content:
            await update.message.reply_text("❌ Sorry, there was an error generating content. Please try again with a different topic.")
            typing_task.cancel()
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = safe_filename(topic)
        
        # Send main text
        await update.message.reply_text(f"📖 **Main Text:**\n\n{content['main_text']}", parse_mode='Markdown')
        
        # Validate vocabulary items are from main text
        main_text = content['main_text']
        missing_vocab = []
        for item in content['vocabulary']:
            if item['chinese'] not in main_text:
                missing_vocab.append(item['chinese'])
        
        if missing_vocab:
            await update.message.reply_text(
                f"⚠️ Note: {len(missing_vocab)} vocabulary items may not be from the main text. "
                f"This sometimes happens with AI generation."
            )
        
        # Create vocabulary file with TTS
        await update.message.reply_text("⏳ Step 2/4: Generating TTS audio for vocabulary items...")
        vocab_filename, vocab_content, audio_files = create_vocabulary_file_with_tts(
            content['vocabulary'], 
            safe_topic
        )
        
        if not audio_files:
            await update.message.reply_text("⚠️ Warning: Could not generate TTS audio for vocabulary. Continuing without audio...")
        
        # Create ZIP package
        await update.message.reply_text("⏳ Step 3/4: Creating Anki import package...")
        zip_filename, zip_buffer = create_zip_package(
            vocab_filename, 
            vocab_content, 
            audio_files, 
            safe_topic, 
            timestamp
        )
        
        # Send ZIP file
        zip_buffer.name = zip_filename
        await update.message.reply_document(
            document=zip_buffer, 
            filename=zip_filename,
            caption="📦 Anki Import Package\n\n"
                   "Contains:\n"
                   f"• {vocab_filename} (vocabulary with TTS tags)\n"
                   f"• {len(audio_files)} MP3 audio files\n\n"
                   "Extract MP3s to your Anki media folder, then import the .txt file!"
        )
        
        # Generate and send opinion texts with audio
        await update.message.reply_text("⏳ Step 4/4: Generating opinion texts with audio...")
        perspectives = [
            ("positive", "Positive View", "😊"),
            ("negative", "Critical View", "🤔"),
            ("balanced", "Balanced View", "⚖️")
        ]
        
        for key, name, emoji in perspectives:
            opinion_text = content['opinion_texts'][key]
            
            # Send text
            await update.message.reply_text(f"{emoji} **{name}:**\n\n{opinion_text}", parse_mode='Markdown')
            
            # Generate and send audio
            await update.message.chat.send_action(action="record_audio")
            audio_data = generate_tts_chirp(opinion_text)
            
            if audio_data:
                audio_filename = f"{safe_topic}_{timestamp}_{key}.mp3"
                audio_file = BytesIO(audio_data)
                audio_file.name = audio_filename
                await update.message.reply_audio(audio=audio_file, filename=audio_filename)
            else:
                await update.message.reply_text(f"⚠️ Could not generate audio for {name}. Text is still available above.")
        
        # Send discussion questions
        questions_text = "💬 **Discussion Questions:**\n\n"
        for i, question in enumerate(content['discussion_questions'], 1):
            questions_text += f"{i}. {question}\n"
        
        await update.message.reply_text(questions_text, parse_mode='Markdown')
        
        await update.message.reply_text(
            "✅ All materials created! Happy learning! 加油！\n\n"
            "📝 To use with Anki:\n"
            "1. Download and extract the ZIP file\n"
            "2. Copy all .mp3 files to your Anki media folder\n"
            "3. Import the .txt file into Anki"
        )
        
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}\n\nPlease try again with a different topic.")
        print(f"Error: {str(e)}")
    finally:
        typing_task.cancel()

def main():
    """Start the bot"""
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    if not DEEPSEEK_API_KEY:
        print("Error: DEEPSEEK_API_KEY not found in environment variables")
        return
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_topic))
    
    # Run the bot
    print("Bot is running...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
