# IMPORTANT: eventlet monkey patch must be FIRST
import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
import asyncio
import json
import base64
import io
import wave
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime
import uuid
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import librosa
import tempfile
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import functools
import webrtcvad
import wave
from pydub import AudioSegment

# Qwen Omni integration
class QwenOmniClient:
    def __init__(self, model_path="Qwen/Qwen2.5-Omni-7B"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the Qwen Omni model and processor"""
        try:
            logger.info("Loading Qwen Omni model...")
            
            # Load model first
            self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
                local_files_only=True  # Force offline mode
            )
            
            # Load processor with offline mode
            self.processor = Qwen2_5OmniProcessor.from_pretrained(
                self.model_path,
                local_files_only=True,  # Force offline mode
                use_fast=False  # Disable fast tokenizer which might require online access
            )
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model loaded on GPU")
            else:
                logger.info("Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Error loading Qwen Omni model: {e}")
            # Try fallback approach
            try:
                logger.info("Trying fallback initialization...")
                self._initialize_model_fallback()
            except Exception as fallback_error:
                logger.error(f"Fallback initialization also failed: {fallback_error}")
                raise e
    
    def _initialize_model_fallback(self):
        """Fallback initialization method"""
        # Set offline mode environment variables
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
        
        # Load model
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load processor
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info("Model loaded on GPU (fallback)")
        else:
            logger.info("Model loaded on CPU (fallback)")
    
    def _save_audio_to_temp_file(self, audio_data: bytes) -> str:
        """Save audio bytes to a temporary file and return the path"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_data)
            return temp_file.name
    
    def _preprocess_audio(self, audio_data: bytes) -> np.ndarray:
        """Preprocess audio data to the required format"""
        try:
            # Save audio data to temporary file
            temp_path = self._save_audio_to_temp_file(audio_data)
            
            # Load audio using librosa
            audio, sr = librosa.load(temp_path, sr=16000)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return audio
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise e
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech to text using Qwen Omni
        """
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_data)
            
            # Create messages for speech-to-text
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are a speech recognition model. Transcribe the audio accurately."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": "Please transcribe this audio into text."},
                ]},
            ]
            
            # Run inference
            result = self._run_inference(messages, audio_array=audio)
            return result
            
        except Exception as e:
            logger.error(f"Error in speech_to_text: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    def generate_response(self, text: str, conversation_history: List[Dict]) -> str:
        """
        Generate response using Qwen Omni LLM
        """
        try:
            # Create system message
            system_prompt = "You are Qwen, a helpful AI assistant created by Alibaba Cloud. You are knowledgeable, helpful, and honest."
            
            # Build conversation messages
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
            ]
            
            # Add conversation history (last 10 messages to avoid context overflow)
            recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
            
            for msg in recent_history[:-1]:  # Exclude the last message as it's the current input
                messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}]
                })
            
            # Add current user message
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": text}]
            })
            
            # Run inference
            result = self._run_inference(messages)
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_response: {e}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    def _run_inference(self, messages: List[Dict], audio_array: np.ndarray = None) -> str:
        """
        Run inference with the Qwen Omni model
        """
        try:
            # Apply chat template with error handling
            try:
                text = self.processor.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as template_error:
                logger.warning(f"Chat template error: {template_error}, using fallback")
                # Fallback: manually construct the prompt
                text = self._construct_prompt_manually(messages)
            
            # Process multimedia information
            try:
                audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
            except Exception as mm_error:
                logger.warning(f"MM info processing error: {mm_error}, using simple approach")
                audios, images, videos = None, None, None
            
            # If we have audio_array, use it instead
            if audio_array is not None:
                audios = [audio_array] if audios is None or len(audios) == 0 else audios
            
            # Process inputs
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=True if audios else False
            )
            
            # Move inputs to model device
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            
            # Generate response
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, 
                    use_audio_in_video=True if audios else False, 
                    return_audio=False, 
                    thinker_max_new_tokens=256, 
                    thinker_do_sample=False,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode output
            response = self.processor.batch_decode(
                output, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Extract the response (remove the input part)
            full_response = response[0]
            
            # Try to find the assistant's response part
            if "<|im_start|>assistant\n" in full_response:
                assistant_response = full_response.split("<|im_start|>assistant\n")[-1]
                if "<|im_end|>" in assistant_response:
                    assistant_response = assistant_response.split("<|im_end|>")[0]
                return assistant_response.strip()
            elif "assistant\n" in full_response:
                assistant_response = full_response.split("assistant\n")[-1]
                return assistant_response.strip()
            else:
                # Fallback: return the generated part after removing input
                input_text = text
                if input_text in full_response:
                    generated_part = full_response.replace(input_text, "").strip()
                    return generated_part if generated_part else full_response
                return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error in _run_inference: {e}")
            raise e
    
    def _construct_prompt_manually(self, messages: List[Dict]) -> str:
        """Manually construct prompt when chat template fails"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if isinstance(content, list):
                text_content = ""
                for item in content:
                    if item.get("type") == "text":
                        text_content += item.get("text", "")
                prompt += f"<|im_start|>{role}\n{text_content}<|im_end|>\n"
            else:
                prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        prompt += "<|im_start|>assistant\n"
        return prompt

# Set offline mode environment variables at startup
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize SocketIO with CORS disabled (allow all origins)
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # Allow all origins
    async_mode='eventlet',  # Use eventlet instead of threading
    logger=True,
    engineio_logger=True
)

# Enable CORS for regular HTTP routes (allow all origins)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for running async operations
executor = ThreadPoolExecutor(max_workers=4)

# Initialize Qwen Omni client (this may take some time)
logger.info("Initializing Qwen Omni client...")
try:
    qwen_client = QwenOmniClient()
    logger.info("Qwen Omni client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qwen Omni client: {e}")
    qwen_client = None

# Store active connections and conversation history
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, str] = {}  # client_id -> session_id
        self.conversation_history: Dict[str, List[Dict]] = {}
    
    def connect(self, client_id: str, session_id: str):
        self.active_connections[client_id] = session_id
        if client_id not in self.conversation_history:
            self.conversation_history[client_id] = []
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections

manager = ConnectionManager()

def process_audio_data(audio_base64: str) -> bytes:
    """Convert base64 audio data to bytes"""
    try:
        # Remove data URL prefix if present
        if "data:audio" in audio_base64:
            audio_base64 = audio_base64.split(",")[1]
        
        audio_data = base64.b64decode(audio_base64)
        return audio_data
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        raise ValueError("Invalid audio data")

def run_in_executor(func):
    """Decorator to run synchronous functions in thread pool"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return executor.submit(func, *args, **kwargs).result()
    return wrapper

# HTTP Routes
@app.route('/')
def root():
    return jsonify({
        "message": "Qwen Omni AI Assistant Backend", 
        "status": "running",
        "model_loaded": qwen_client is not None
    })

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "model_status": "loaded" if qwen_client is not None else "not_loaded",
        "gpu_available": torch.cuda.is_available()
    })

@app.route('/conversation/<client_id>')
def get_conversation_history(client_id):
    """Get conversation history for a client"""
    if client_id in manager.conversation_history:
        return jsonify({"history": manager.conversation_history[client_id]})
    return jsonify({"history": []})

@app.route('/conversation/<client_id>', methods=['DELETE'])
def clear_conversation_history(client_id):
    """Clear conversation history for a client"""
    if client_id in manager.conversation_history:
        manager.conversation_history[client_id] = []
        return jsonify({"message": "Conversation history cleared"})
    return jsonify({"message": "Client not found"})

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    client_id = request.args.get('client_id')
    if not client_id:
        disconnect()
        return False
    
    if qwen_client is None:
        emit('error', {'message': 'Model not loaded'})
        disconnect()
        return False
    
    session_id = request.sid
    manager.connect(client_id, session_id)
    logger.info(f"Client {client_id} connected with session {session_id}")
    
    emit('connected', {'message': f'Connected as {client_id}'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = None
    # Find client_id by session_id
    for cid, sid in manager.active_connections.items():
        if sid == request.sid:
            client_id = cid
            break
    
    if client_id:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")

def process_audio_async(client_id, audio_data):
    """Process audio in a separate thread"""
    try:
        # Send processing status
        socketio.emit('status', {
            'message': 'Processing audio...',
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
        # Convert speech to text
        transcribed_text = qwen_client.speech_to_text(audio_data)
        
        # Send transcription
        socketio.emit('transcription', {
            'text': transcribed_text,
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
        # Add to conversation history
        manager.conversation_history[client_id].append({
            "role": "user",
            "content": transcribed_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response
        ai_response = qwen_client.generate_response(
            transcribed_text, 
            manager.conversation_history[client_id]
        )
        
        # Add AI response to history
        manager.conversation_history[client_id].append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send AI response
        socketio.emit('response', {
            'text': ai_response,
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
    except Exception as e:
        logger.error(f"Error processing audio for client {client_id}: {e}")
        socketio.emit('error', {
            'message': f'Error processing audio: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])

def process_text_async(client_id, user_text):
    """Process text in a separate thread"""
    try:
        # Add to conversation history
        manager.conversation_history[client_id].append({
            "role": "user",
            "content": user_text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate AI response
        ai_response = qwen_client.generate_response(
            user_text, 
            manager.conversation_history[client_id]
        )
        
        # Add AI response to history
        manager.conversation_history[client_id].append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Send AI response
        socketio.emit('response', {
            'text': ai_response,
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])
        
    except Exception as e:
        logger.error(f"Error processing text for client {client_id}: {e}")
        socketio.emit('error', {
            'message': f'Error processing text: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }, room=manager.active_connections[client_id])

# convert audio formats
def convert_webm_to_wav(audio_bytes: bytes) -> bytes:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    out_io = io.BytesIO()
    audio.export(out_io, format="wav")
    return out_io.getvalue()

def contains_speech(wav_bytes: bytes) -> bool:
    vad = webrtcvad.Vad(2)  # Aggressiveness: 0–3
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        sample_rate = wf.getframerate()
        frame_duration = 30  # ms
        frame_size = int(sample_rate * 2 * frame_duration / 1000)
        frames = wf.readframes(wf.getnframes())

        for i in range(0, len(frames), frame_size):
            chunk = frames[i:i+frame_size]
            if len(chunk) < frame_size:
                break
            if vad.is_speech(chunk, sample_rate):
                return True
    return False

# @socketio.on('audio')
# def handle_audio(data):
#     client_id = data.get('client_id')
#     audio_base64 = data.get('audio')
    
#     if not client_id or not manager.is_connected(client_id):
#         emit('error', {'message': 'Client not connected'})
#         return
    
#     try:
#         audio_data = process_audio_data(audio_base64)
#         # Use eventlet spawn for async processing
#         eventlet.spawn(process_audio_async, client_id, audio_data)
        
#     except Exception as e:
#         logger.error(f"Error handling audio for client {client_id}: {e}")
#         emit('error', {
#             'message': f'Error processing audio: {str(e)}',
#             'timestamp': datetime.now().isoformat()
#         })

@socketio.on('audio')
def handle_audio(data):
    client_id = data.get('client_id')
    audio_base64 = data.get('audio')
    
    if not client_id or not manager.is_connected(client_id):
        emit('error', {'message': 'Client not connected'})
        return
    
    try:
        audio_data = process_audio_data(audio_base64)  # bytes of webm
        wav_data = convert_webm_to_wav(audio_data)  # convert to WAV
        
        if not contains_speech(wav_data):
            emit('info', {'message': 'Audio contains no speech'})
            return
        
        # Run async processing if speech detected
        eventlet.spawn(process_audio_async, client_id, audio_data)
        
    except Exception as e:
        logger.error(f"Error handling audio for client {client_id}: {e}")
        emit('error', {
            'message': f'Error processing audio: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })


@socketio.on('text')
def handle_text(data):
    client_id = data.get('client_id')
    user_text = data.get('text')
    
    if not client_id or not manager.is_connected(client_id):
        emit('error', {'message': 'Client not connected'})
        return
    
    if not user_text:
        emit('error', {'message': 'No text provided'})
        return
    
    try:
        # Use eventlet spawn for async processing
        eventlet.spawn(process_text_async, client_id, user_text)
        
    except Exception as e:
        logger.error(f"Error handling text for client {client_id}: {e}")
        emit('error', {
            'message': f'Error processing text: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

@socketio.on('ping')
def handle_ping():
    emit('pong')

if __name__ == '__main__':
    # Run the Flask-SocketIO app with eventlet
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=8010, 
        debug=False,
        use_reloader=False,  # Disable reloader to prevent model loading twice
        log_output=True
    )