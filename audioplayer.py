import time
import threading
import speech_recognition as sr
from openai import OpenAI
import os
import subprocess
from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import asyncio
import aiofiles
import pygame

app = Flask(__name__)
socketio = SocketIO(app)

client = OpenAI()

audiobook_path = "output_small.mp3"
N_MINUTES = 10
playback_active = False
current_position = 0  # Track playback position
current_speed = 1.0

BOOKMARKS_FILE = "bookmarks.json"

SPEED_FILES = {
    1.0: "output_small.mp3",
    1.25: "output_small_1.25x.mp3",
    1.5: "output_small_1.5x.mp3"
}

# Add to globals
speed_files_ready = {
    1.0: True,  # Base file is always ready
    1.25: False,
    1.5: False
}

# Add a global variable to control listening
listening_event = None

def generate_speed_file(speed_info):
    """Generate a single speed-adjusted version of the audiobook."""
    global speed_files_ready
    speed, filepath = speed_info
    if not Path(filepath).exists():
        print(f"Generating {speed}x speed version...")
        subprocess.run([
            "ffmpeg", "-y", "-i", SPEED_FILES[1.0],
            "-filter:a", f"atempo={speed}",
            "-vn", filepath
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    speed_files_ready[speed] = True
    socketio.emit('speed_file_ready', {'speed': speed})
    return filepath

def check_speed_files_async():
    """Check and generate speed files in background."""
    base_file = SPEED_FILES[1.0]
    if not Path(base_file).exists():
        print("Error: Base audio file not found")
        return False
    
    # Check which files exist already
    for speed, filepath in SPEED_FILES.items():
        if speed != 1.0:
            speed_files_ready[speed] = Path(filepath).exists()
    
    # Find which speed files need to be generated
    missing_speeds = [(speed, filepath) 
                     for speed, filepath in SPEED_FILES.items() 
                     if speed != 1.0 and not Path(filepath).exists()]
    
    if missing_speeds:
        print(f"Generating {len(missing_speeds)} missing speed variants in background...")
        socketio.emit('status_update', {'message': 'Generating speed variants in background...'})
        
        def generate_in_background():
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(generate_speed_file, missing_speeds))
            socketio.emit('status_update', {'message': 'All speed variants ready!'})
        
        thread = threading.Thread(target=generate_in_background)
        thread.start()
    
    return True

def play_audio():
    """Plays the audiobook from the last saved position."""
    global playback_active, current_position
    
    # Ensure mixer is initialized
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    
    # Calculate position for different speeds
    real_position = current_position / current_speed
    
    pygame.mixer.music.load(SPEED_FILES[current_speed])
    pygame.mixer.music.play(start=int(real_position))
    playback_active = True
    
    duration = get_audio_duration(SPEED_FILES[current_speed])
    
    while playback_active and pygame.mixer.music.get_busy():
        if playback_active:
            current_time = real_position + (pygame.mixer.music.get_pos() / 1000)
            # Adjust current time for speed
            adjusted_time = current_time * current_speed
            socketio.emit('time_update', {
                'current_time': adjusted_time,
                'duration': duration,
                'speed': current_speed
            })
        time.sleep(0.1)

def stop_audio():
    """Stops the audiobook and saves the current position."""
    global playback_active, current_position
    try:
        if pygame.mixer.music.get_busy():
            current_position += pygame.mixer.music.get_pos() / 1000  # Convert ms to seconds
    except pygame.error:
        # Re-initialize mixer if it's not initialized
        pygame.mixer.init()
    
    playback_active = False
    pygame.mixer.music.stop()

def get_audio_duration(filepath):
    """Returns the total duration of an audio file in seconds."""
    result = subprocess.run(
        ["ffmpeg", "-i", filepath],
        stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            duration_str = line.split(",")[0].split("Duration:")[1].strip()
            h, m, s = map(float, duration_str.split(":"))
            return int(h * 3600 + m * 60 + s)
    
    return None  # If parsing fails

def transcribe_audio(duration):
    """Extracts the last N minutes of audio and transcribes it."""
    temp_audio_path = "temp_clip.mp3"
    
    # Get total audiobook duration
    total_seconds = get_audio_duration(audiobook_path)
    
    # Ensure valid segment
    start_time = max(0, current_position - (duration * 60))  

    # Extract last N minutes
    subprocess.run([
        "ffmpeg", "-y", "-i", audiobook_path, "-ss", str(start_time), "-t", str(duration * 60),
        "-acodec", "copy", temp_audio_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # Silence ffmpeg output

    with open(temp_audio_path, "rb") as file:
        return client.audio.transcriptions.create(model="whisper-1", file=file).text

def transcribe_user_question():
    """Records and transcribes the user's spoken question."""
    global listening_event
    recognizer = sr.Recognizer()
    listening_event = threading.Event()

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=None)
            # Process and show question immediately
            temp_audio_path = "user_question.wav"
            with open(temp_audio_path, "wb") as file:
                file.write(audio.get_wav_data())

            with open(temp_audio_path, "rb") as file:
                question = client.audio.transcriptions.create(model="whisper-1", file=file).text
                # Emit the question immediately
                socketio.emit('status_update', {'message': f'Your question: "{question}"'})
                return question
        except sr.WaitTimeoutError:
            return "No question was asked"

def query_llm(audiobook_text, user_question):
    """Sends audiobook context and user's question to LLM."""
    print("TEXT", audiobook_text)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": 
             "You are Audiobook Copilot, an AI companion that helps listeners understand and engage with audiobooks. Respond concisely while providing helpful, accurate answers."},
            {"role": "user", "content": f"Context: {audiobook_text}\nQuestion: {user_question}"}
        ],
        temperature=0.7, max_tokens=200
    )
    return response.choices[0].message.content

def generate_speech(text):
    """Converts LLM's response into speech and saves it."""
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    
    response_audio_path = "response.mp3"
    response_fast_path = "response_fast.mp3"
    
    # Save original response
    with open(response_audio_path, "wb") as file:
        file.write(response.content)
    
    # Create 1.25x speed version
    subprocess.run([
        "ffmpeg", "-y", "-i", response_audio_path,
        "-filter:a", "atempo=1.25",
        "-vn", response_fast_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Tell client to play the response with a timestamp to prevent caching
    socketio.emit('play_response', {'timestamp': time.time()})

def jump_to(timestamp):
    """Update current position."""
    global current_position
    current_position = timestamp

def load_bookmarks():
    """Load bookmarks from JSON file."""
    if not os.path.exists(BOOKMARKS_FILE):
        return {"bookmarks": []}
    try:
        with open(BOOKMARKS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"bookmarks": []}

def save_bookmarks(bookmarks):
    """Save bookmarks to JSON file."""
    with open(BOOKMARKS_FILE, 'w') as f:
        json.dump(bookmarks, f, indent=4)

async def transcribe_chunk_async(start_time, duration):
    """Asynchronously transcribe a small chunk of audio."""
    temp_chunk = f"temp_chunk_{start_time}.mp3"
    
    # Extract chunk using ffmpeg
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", audiobook_path,
        "-ss", str(start_time),
        "-t", str(duration),
        "-acodec", "copy", temp_chunk,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await proc.communicate()
    
    # Transcribe chunk using OpenAI API
    loop = asyncio.get_event_loop()
    async with aiofiles.open(temp_chunk, "rb") as f:
        content = await f.read()
        response = await loop.run_in_executor(
            None,
            lambda: client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.mp3", content),
                response_format="json"
            )
        )
    
    # Cleanup temp file
    try:
        os.remove(temp_chunk)
    except:
        pass
    
    # Create a segment with the chunk's text and timing
    segment = {
        "text": response.text,
        "start": start_time,
        "end": start_time + duration
    }
    
    return [segment]  # Return as list for consistency

async def generate_full_transcript_async():
    """Generate transcript using multiple small chunks in parallel."""
    CHUNK_DURATION = 30  # 30 second chunks for better granularity
    MAX_CONCURRENT = 3   # Number of concurrent transcriptions
    
    # Get total duration
    total_duration = get_audio_duration(audiobook_path)
    total_chunks = (total_duration + CHUNK_DURATION - 1) // CHUNK_DURATION
    
    print(f"Starting transcript generation: {total_duration}s in {total_chunks} chunks")
    socketio.emit('status_update', 
                 {'message': f'Starting transcript generation (0/{total_chunks} chunks)'})
    
    all_segments = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    async def process_chunk(chunk_num):
        async with semaphore:
            start_time = chunk_num * CHUNK_DURATION
            current_duration = min(CHUNK_DURATION, total_duration - start_time)
            
            print(f"Processing chunk {chunk_num+1}/{total_chunks} ({start_time}-{start_time+current_duration}s)")
            socketio.emit('status_update', 
                         {'message': f'Generating transcript ({chunk_num+1}/{total_chunks} chunks)'})
            
            try:
                segments = await transcribe_chunk_async(start_time, current_duration)
                print(f"Successfully processed chunk {chunk_num+1}")
                return segments
            except Exception as e:
                print(f"Error processing chunk {chunk_num}: {str(e)}")
                return []
    
    # Process chunks in parallel with semaphore limiting concurrency
    tasks = [process_chunk(i) for i in range(total_chunks)]
    chunk_results = await asyncio.gather(*tasks)
    
    # Combine and sort all segments
    for segments in chunk_results:
        all_segments.extend(segments)
    all_segments.sort(key=lambda x: x["start"])
    
    # Save complete transcript
    async with aiofiles.open("transcript.json", "w") as f:
        await f.write(json.dumps({"segments": all_segments}, indent=4))
    
    print("Transcript generation complete!")
    socketio.emit('status_update', {'message': 'Transcript generation complete!'})

def start_transcript_generation():
    """Start async transcript generation in the background."""
    async def run():
        await generate_full_transcript_async()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run())

def get_context_at_position(position, minutes=10):
    """Get N minutes of transcript context before the given position."""
    try:
        with open("transcript.json", "r") as f:
            transcript = json.load(f)
        
        # Find segments within our time window
        start_time = max(0, position - (minutes * 60))
        end_time = position  # Current position
        
        # Debug prints
        print(f"Looking for context between {start_time}s and {end_time}s")
        
        relevant_segments = []
        for seg in transcript["segments"]:
            if start_time <= seg["start"] <= end_time:
                relevant_segments.append(seg["text"])
                print(f"Found segment: {seg['text'][:50]}...")
        
        if not relevant_segments:
            print("No segments found in time window!")
            # Expand search if no segments found
            for seg in transcript["segments"]:
                if abs(seg["start"] - position) < 60:  # Look within 1 minute
                    relevant_segments.append(seg["text"])
                    print(f"Found nearby segment: {seg['text'][:50]}...")
        
        context = " ".join(relevant_segments)
        print(f"Total context length: {len(context)} characters")
        return context
        
    except Exception as e:
        print(f"Error getting context: {str(e)}")
        return "Error getting audiobook context."

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('play')
def handle_play(data):
    global playback_active
    playback_active = True
    return {'status': 'playing', 'position': current_position}

@socketio.on('pause')
def handle_pause(data):
    global playback_active
    playback_active = False
    return {'status': 'paused', 'position': current_position}

@socketio.on('ask_question')
def handle_question(data):
    global playback_active
    current_position = data.get('current_time', 0)  # Get position from client
    
    socketio.emit('status_update', {'message': 'Processing...'})
    
    # Get context from pre-generated transcript
    audiobook_text = get_context_at_position(current_position, N_MINUTES)
    
    # Now just handle the question
    socketio.emit('status_update', {'message': 'Listening for your question... (speak now)'})
    socketio.emit('start_listening')
    user_question = transcribe_user_question()
    
    socketio.emit('stop_listening')
    socketio.emit('status_update', {'message': 'Getting answer...'})
    
    answer = query_llm(audiobook_text, user_question)
    print(f"AI Answer: {answer}")
    socketio.emit('status_update', {'message': 'Playing answer...'})
    generate_speech(answer)
    
    return {'status': 'questioning'}

@socketio.on('continue')
def handle_continue(data):
    global playback_active
    if not playback_active:
        playback_active = True
    return {'status': 'playing', 'position': current_position}

@socketio.on('quit')
def handle_quit(data):
    playback_active = False
    return {'status': 'stopped'}

@socketio.on('jump_to')
def handle_jump(data):
    global playback_active
    timestamp = data.get('timestamp', 0)
    current_position = timestamp
    return {'status': 'playing', 'position': current_position}

@socketio.on('add_bookmark')
def handle_add_bookmark(data):
    bookmarks = load_bookmarks()
    new_bookmark = {
        "title": data['title'],
        "timestamp": data['timestamp']
    }
    bookmarks['bookmarks'].append(new_bookmark)
    save_bookmarks(bookmarks)
    return {'status': 'success', 'bookmarks': bookmarks['bookmarks']}

@socketio.on('delete_bookmark')
def handle_delete_bookmark(data):
    bookmarks = load_bookmarks()
    bookmarks['bookmarks'] = [b for b in bookmarks['bookmarks'] 
                             if b['timestamp'] != data['timestamp']]
    save_bookmarks(bookmarks)
    return {'status': 'success', 'bookmarks': bookmarks['bookmarks']}

@socketio.on('get_bookmarks')
def handle_get_bookmarks(data):
    bookmarks = load_bookmarks()
    return {'status': 'success', 'bookmarks': bookmarks['bookmarks']}

@socketio.on('update_bookmark')
def handle_update_bookmark(data):
    bookmarks = load_bookmarks()
    for bookmark in bookmarks['bookmarks']:
        if bookmark['timestamp'] == data['old_timestamp']:
            bookmark['title'] = data['title']
            bookmark['timestamp'] = data['new_timestamp']
    save_bookmarks(bookmarks)
    return {'status': 'success', 'bookmarks': bookmarks['bookmarks']}

@socketio.on('set_speed')
def handle_set_speed(data):
    global current_speed
    current_speed = float(data.get('speed', 1.0))
    return {'status': 'success', 'speed': current_speed}

@socketio.on('get_speed_status')
def handle_get_speed_status(data):
    return {'status': 'success', 'ready': speed_files_ready}

@socketio.on('stop_question')
def handle_stop_question(data):
    """Handle request to stop listening for question."""
    global listening_event
    if listening_event:
        listening_event.set()
        socketio.emit('status_update', {'message': 'Processing audiobook context...'})
    return {'status': 'success'}

@app.route('/audio/<filename>')
def serve_audio(filename):
    return send_file(filename, mimetype='audio/mpeg')

@app.route('/response_audio')
def serve_response():
    # Add cache control headers to prevent caching
    response = send_file('response_fast.mp3', mimetype='audio/mpeg')
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    if not Path(SPEED_FILES[1.0]).exists():
        print("Error: Base audio file not found")
        exit(1)
    
    # Start the server first
    print("Starting server...")
    
    # Check/start transcript generation in background
    if not Path("transcript.json").exists():
        print("Transcript not found, will generate in background...")
        transcript_thread = threading.Thread(target=start_transcript_generation)
        transcript_thread.daemon = True
        transcript_thread.start()
    
    # Start speed file generation in background
    check_speed_files_async()
    
    # Start server with host='0.0.0.0' to make it accessible externally
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
