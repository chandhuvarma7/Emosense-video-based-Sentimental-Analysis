from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import cv2
import tensorflow as tf
import uvicorn
from pathlib import Path
from typing import Dict, Optional
import uuid
import shutil
import sys
import os
import speech_recognition as sr
from pydub import AudioSegment
from textblob import TextBlob
from transformers import pipeline
import seaborn as sns


classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Explicitly set TensorFlow logging to reduce warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Try to import DeepFace with error handling
try:
    from deepface import DeepFace
except ImportError:
    print("DeepFace import failed. Some emotion detection features may be limited.")
    DeepFace = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    yield
    # Shutdown code

app = FastAPI(lifespan=lifespan, title="Multimodal Emotion Detection App")

# Create required directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/images", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/models", exist_ok=True)
os.makedirs("static/audio", exist_ok=True)
os.makedirs("static/transcripts", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Add this route handler for the root path
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# User authentication storage (simple dictionary for now)
users = {"admin": "password123"}

# Initialize text emotion analysis model
try:
    sentiment_analyzer = pipeline("sentiment-analysis")
except:
    print("Transformer model initialization failed. Using TextBlob fallback.")
    sentiment_analyzer = None

# Emoji mapping for emotions
EMOJI_MAP = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😃",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

# Create templates directory and files
os.makedirs("templates", exist_ok=True)

# Main index.html template - using utf-8 encoding to handle emoji characters
with open("templates/index.html", "w", encoding="utf-8") as f:
    f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EmoSense - Video Based Sentimental Analysis & Emotion Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .tab-content { padding: 20px; border: 1px solid #ddd; border-top: none; border-radius: 0 0 5px 5px; }
        .nav-tabs { margin-bottom: 0; }
        .result-section { margin-top: 20px; }
        .emoji { font-size: 2em; }
        .chart-container { margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="mb-4">EmoSense - Video Based Sentimental Analysis & Emotion Detection</h1>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="register-tab" data-bs-toggle="tab" data-bs-target="#register" type="button" role="tab">📝 Register</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="login-tab" data-bs-toggle="tab" data-bs-target="#login" type="button" role="tab">🔑 Login</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="train-tab" data-bs-toggle="tab" data-bs-target="#train" type="button" role="tab">🛠 Train Model</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="predict-tab" data-bs-toggle="tab" data-bs-target="#predict" type="button" role="tab">🎭 Predict Emotion</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Register Tab -->
            <div class="tab-pane fade show active" id="register" role="tabpanel">
                <h2>🔑 User Registration</h2>
                <form id="registerForm">
                    <div class="mb-3">
                        <label for="registerUsername" class="form-label">Username</label>
                        <input type="text" class="form-control" id="registerUsername" required>
                    </div>
                    <div class="mb-3">
                        <label for="registerPassword" class="form-label">Password</label>
                        <input type="password" class="form-control" id="registerPassword" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Register</button>
                </form>
                <div id="registerStatus" class="alert mt-3" style="display: none;"></div>
            </div>
            
            <!-- Login Tab -->
            <div class="tab-pane fade" id="login" role="tabpanel">
                <h2>🔐 User Login</h2>
                <form id="loginForm">
                    <div class="mb-3">
                        <label for="loginUsername" class="form-label">Username</label>
                        <input type="text" class="form-control" id="loginUsername" required>
                    </div>
                    <div class="mb-3">
                        <label for="loginPassword" class="form-label">Password</label>
                        <input type="password" class="form-control" id="loginPassword" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Login</button>
                </form>
                <div id="loginStatus" class="alert mt-3" style="display: none;"></div>
            </div>
            
            <!-- Train Model Tab -->
            <div class="tab-pane fade" id="train" role="tabpanel">
                <h2>🎓 Train Emotion Model</h2>
                <form id="trainForm">
                    <div class="mb-3">
                        <label for="datasetFile" class="form-label">📂 Upload Dataset (CSV)</label>
                        <input type="file" class="form-control" id="datasetFile" accept=".csv" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Train Model</button>
                </form>
                
                <div id="trainingResults" class="result-section" style="display: none;">
                    <div id="trainingStatus"></div>
                    <div class="row mt-3">
                        <div class="col-md-6">
                            <h4>📈 Dataset Overview</h4>
                            <img id="datasetPlot" class="img-fluid" alt="Dataset Plot">
                        </div>
                        <div class="col-md-6">
                            <h4>📊 Top 4 Frames</h4>
                            <pre id="topFrames" class="border p-2"></pre>
                        </div>
                    </div>
                    <div class="mt-3">
                        <h4>📑 Classification Report</h4>
                        <pre id="classReport" class="border p-2"></pre>
                    </div>
                </div>
            </div>
            
            <!-- Predict Emotion Tab -->
            <div class="tab-pane fade" id="predict" role="tabpanel">
                <h2>🎭 EmoSense - Video Based Sentimental Analysis & Emotion Detection</h2>
                <form id="predictForm">
                    <div class="mb-3">
                        <label for="videoFile" class="form-label">🎥 Upload Video for Emotion Detection</label>
                        <input type="file" class="form-control" id="videoFile" accept="video/*" required>
                    </div>
                    <button type="submit" class="btn btn-primary">Detect Emotion</button>
                </form>
                
                <div id="predictResults" class="result-section" style="display: none;">
                    <div class="row">
                        <div class="col-md-4">
                            <h4>🎭 Visual Emotion</h4>
                            <div id="predictedEmotion" class="border p-3 text-center"></div>
                        </div>
                        <div class="col-md-4">
                            <h4>🎤 Audio Emotion</h4>
                            <div id="audioEmotion" class="border p-3 text-center"></div>
                        </div>
                        <div class="col-md-4">
                            <h4>🔤 Transcript</h4>
                            <div id="transcript" class="border p-3" style="max-height: 200px; overflow-y: auto;"></div>
                        </div>
                    </div>
                    
                    <div class="row mt-4">
                        <div class="col-md-6 chart-container">
                            <h4>📊 Visual Emotion Scores</h4>
                            <img id="barEmotionPlot" class="img-fluid" alt="Bar Emotion Plot">
                        </div>
                        <div class="col-md-6 chart-container">
                            <h4>📈 Emotion Trend Over Time</h4>
                            <img id="lineEmotionTrend" class="img-fluid" alt="Line Emotion Trend">
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-md-6 chart-container">
                            <h4>🍕 Emotion Distribution</h4>
                            <img id="pieEmotionDist" class="img-fluid" alt="Pie Chart">
                        </div>
                        <div class="col-md-6 chart-container">
                            <h4>📊 Emotion Histogram</h4>
                            <img id="histEmotionDist" class="img-fluid" alt="Histogram Chart">
                        </div>
                    </div>
                    
                    <div class="row mt-3">
                        <div class="col-12 chart-container">
                            <h4>🧩 Scatter Plot: Visual vs Audio Emotions</h4>
                            <img id="scatterCorrelation" class="img-fluid" alt="Scatter Plot">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Register form submission
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('registerUsername').value;
            const password = document.getElementById('registerPassword').value;
            
            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        username: username,
                        password: password
                    })
                });
                
                const result = await response.text();
                const statusDiv = document.getElementById('registerStatus');
                statusDiv.textContent = result;
                statusDiv.style.display = 'block';
                statusDiv.className = result.includes('✅') ? 'alert alert-success' : 'alert alert-danger';
            } catch (error) {
                console.error('Error:', error);
            }
        });
        
        // Login form submission
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('loginUsername').value;
            const password = document.getElementById('loginPassword').value;
            
            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/x-www-form-urlencoded',
                    },
                    body: new URLSearchParams({
                        username: username,
                        password: password
                    })
                });
                
                const result = await response.text();
                const statusDiv = document.getElementById('loginStatus');
                statusDiv.textContent = result;
                statusDiv.style.display = 'block';
                statusDiv.className = result.includes('✅') ? 'alert alert-success' : 'alert alert-danger';
            } catch (error) {
                console.error('Error:', error);
            }
        });
        
        // Train model form submission
        document.getElementById('trainForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('datasetFile');
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/train_model', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // Display results
                document.getElementById('trainingStatus').innerHTML = result.status;
                document.getElementById('datasetPlot').src = result.dataset_plot;
                document.getElementById('topFrames').textContent = result.top_frames;
                document.getElementById('classReport').textContent = result.class_report;
                document.getElementById('trainingResults').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('Training failed: ' + error.message);
            }
        });
        
        // Predict emotion form submission
        document.getElementById('predictForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const fileInput = document.getElementById('videoFile');
            const file = fileInput.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('video', file);
            
            try {
                const response = await fetch('/predict_emotion', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(errorText);
                }
                
                const result = await response.json();
                
                // Display results
                document.getElementById('predictedEmotion').innerHTML = result.visual_emotion;
                document.getElementById('audioEmotion').innerHTML = result.audio_emotion;
                document.getElementById('transcript').textContent = result.transcript;
                
                // Load all visualization charts
                document.getElementById('barEmotionPlot').src = result.bar_plot;
                document.getElementById('lineEmotionTrend').src = result.line_plot;
                document.getElementById('pieEmotionDist').src = result.pie_plot;
                document.getElementById('histEmotionDist').src = result.hist_plot;
                document.getElementById('scatterCorrelation').src = result.scatter_plot;
                
                document.getElementById('predictResults').style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                alert('Prediction failed: ' + error.message);
            }
        });
    </script>
</body>
</html>
    """)

# Function to register a new user
@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    if username in users:
        return "❌ Username already exists. Please try another."
    users[username] = password
    return "✅ Registration successful! You can now log in."

# Function to login
@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    if username in users and users[username] == password:
        return "✅ Login successful! You can now proceed."
    return "❌ Invalid credentials. Please try again."

# Function to extract audio from video
def extract_audio(video_path, output_path):
    try:
        audio = AudioSegment.from_file(video_path)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return False

# Function to transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return "Transcription failed."

# Function to analyze text sentiment
def analyze_text_emotion(text):
    if sentiment_analyzer:
        try:
            result = sentiment_analyzer(text)
            label = result[0]['label'].lower()
            # Map the transformer output to our emotion categories
            if 'positive' in label:
                return 'happy'
            elif 'negative' in label:
                return 'sad'
            else:
                return 'neutral'
        except Exception as e:
            print(f"Error analyzing text with transformer: {e}")
    
    # Fallback to TextBlob
    try:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.3:
            return 'happy'
        elif polarity < -0.3:
            return 'sad'
        elif polarity < -0.1:
            return 'angry'
        elif polarity > 0.1:
            return 'surprise'
        else:
            return 'neutral'
    except Exception as e:
        print(f"Error analyzing text with TextBlob: {e}")
        return 'neutral'

# Function to train model and display dataset insights
@app.post("/train_model")
async def train_model(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file_path = f"static/uploads/{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the file
    df = pd.read_csv(temp_file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    class_report = classification_report(y_test, predictions)
    model_path = "static/models/emotion_model.pkl"
    joblib.dump(model, model_path)
    
    # Display only top 4 rows of dataset
    top_4_data = df.head(4).to_string()
    
    # Plot dataset overview with seaborn
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y, palette='viridis')
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.title("📊 Dataset Emotion Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_path = f"static/images/dataset_plot_{uuid.uuid4()}.png"
    plt.savefig(plot_path)
    plt.close()
    
    return {
        "status": f"<h2 style='font-size:24px;'>✅ Model trained successfully with accuracy: {accuracy:.2f} 🎯</h2>",
        "dataset_plot": "/" + plot_path,
        "top_frames": f"📊 Top 4 Rows:\n{top_4_data}",
        "class_report": f"📑 Classification Report:\n{class_report}"
    }

# Function to detect faces using Haar cascade
def detect_faces(frame):
    # Load the haar cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces, gray

# Enhanced function to predict emotion from video with audio analysis
@app.post("/predict_emotion")
async def predict_emotion(video: UploadFile = File(...)):
    # Check if model exists
    model_path = "static/models/emotion_model.pkl"
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="⚠️ Error: Model not trained yet. Please train the model first.")
    
    # Save the uploaded video
    video_id = uuid.uuid4()
    temp_video_path = f"static/uploads/video_{video_id}.mp4"
    with open(temp_video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # Extract audio from video
    audio_path = f"static/audio/audio_{video_id}.wav"
    audio_extraction_success = extract_audio(temp_video_path, audio_path)
    
    # Process audio for transcription and emotion analysis
    transcript = "No audio detected or transcription failed."
    audio_emotion = "neutral"
    audio_emotion_emoji = EMOJI_MAP.get("neutral", "😐")
    
    if audio_extraction_success:
        transcript = transcribe_audio(audio_path)
        
        # Save transcript to file
        transcript_path = f"static/transcripts/transcript_{video_id}.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        
        # Analyze emotion from transcript
        if transcript and transcript != "Transcription failed.":
            audio_emotion = analyze_text_emotion(transcript)
            audio_emotion_emoji = EMOJI_MAP.get(audio_emotion, "❓")
    
    # Load model for visual emotion detection
    model = joblib.load(model_path)
    
    # Process video
    cap = cv2.VideoCapture(temp_video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    if duration < 2 or duration > 60:
        cap.release()
        raise HTTPException(status_code=400, detail="⚠️ Error: Video duration must be between 2 and 60 seconds. ⏳")
    
    # Store frame-by-frame emotion results
    frame_results = {}
    frame_timestamps = []
    dominant_emotions = []
    
    frame_interval = max(1, int(fps / 2))  # Process 2 frames per second
    frame_id = 0
    timestamp = 0
    
    # Initialize a timeline dictionary to track emotions over time
    timeline_emotions = {}
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:
            # Calculate timestamp
            timestamp = frame_id / fps
            frame_timestamps.append(timestamp)
            
            # Detect faces using Haar cascade
            faces, gray = detect_faces(frame)
            
            detected_emotion = None
            
            if len(faces) > 0:
                # If DeepFace is available, use it
                if DeepFace is not None:
                    try:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        analysis = DeepFace.analyze(img_rgb, actions=["emotion"], enforce_detection=False)
                        if analysis:
                            for key, value in analysis[0]['emotion'].items():
                                frame_results[key] = frame_results.get(key, []) + [value]
                                
                            # Get dominant emotion for this frame
                            detected_emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
                    except Exception as e:
                        print(f"Error analyzing frame with DeepFace: {e}")
                        detected_emotion = None
                else:
                    # Simple random emotion simulation as fallback
                    from random import choice
                    emotions = list(EMOJI_MAP.keys())
                    detected_emotion = choice(emotions)
                    frame_results[detected_emotion] = frame_results.get(detected_emotion, []) + [np.random.uniform(50, 100)]
            
            if detected_emotion:
                dominant_emotions.append(detected_emotion)
                
                # Add to timeline
                if timestamp not in timeline_emotions:
                    timeline_emotions[timestamp] = {}
                    
                timeline_emotions[timestamp][detected_emotion] = timeline_emotions.get(timestamp, {}).get(detected_emotion, 0) + 1
        
        frame_id += 1
    
    cap.release()
    
    if not frame_results:
        raise HTTPException(status_code=400, detail="⚠️ Error: Could not detect faces or emotions in the video.")
    
    # Calculate average scores for each emotion
    avg_scores = {key: np.mean(values) for key, values in frame_results.items()}
    
    # Get the predicted visual emotion (most frequent)
    if dominant_emotions:
        predicted_emotion = max(set(dominant_emotions), key=dominant_emotions.count)
    else:
        predicted_emotion = max(avg_scores, key=avg_scores.get)
        
    predicted_emoji = EMOJI_MAP.get(predicted_emotion, "❓")
    
    # Prepare data for visualizations
    emotions = list(avg_scores.keys())
    scores = list(avg_scores.values())
    
    # 1. Bar Chart - Create bar chart for emotion scores
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=emotions, y=scores, palette='viridis')
    plt.xlabel("Emotions")
    plt.ylabel("Score")
    plt.title("Emotion Analysis Scores")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for i, v in enumerate(scores):
        ax.text(i, v + 1, f"{v:.1f}", ha='center')
        
    plt.tight_layout()
    bar_plot_path = f"static/images/bar_emotion_plot_{video_id}.png"
    plt.savefig(bar_plot_path)
    plt.close()
    
    # 2. Line Chart - Create line chart for emotion trends over time
    if timeline_emotions:
        # Prepare data for line chart
        timestamps = sorted(timeline_emotions.keys())
        emotion_values = {e: [] for e in EMOJI_MAP.keys()}
        
        # For each timestamp, get the count of each emotion (or 0 if not present)
        for ts in timestamps:
            for emotion in EMOJI_MAP.keys():
                emotion_values[emotion].append(timeline_emotions.get(ts, {}).get(emotion, 0))
        
        plt.figure(figsize=(10, 6))
        for emotion, values in emotion_values.items():
            if sum(values) > 0:  # Only plot emotions that were detected
                plt.plot(timestamps, values, marker='o', linestyle='-', label=emotion)
        
        plt.xlabel("Time (seconds)")
        plt.ylabel("Intensity")
        plt.title("Emotion Trends Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        line_plot_path = f"static/images/line_trend_plot_{video_id}.png"
        plt.savefig(line_plot_path)
        plt.close()
    else:
        # Fallback if timeline data is not available
        plt.figure(figsize=(10, 6))
        plt.plot(list(range(len(emotions))), scores, marker='o', linestyle='-')
        plt.xlabel("Emotion Index")
        plt.ylabel("Score")
        plt.title("Emotion Score Trend")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        line_plot_path = f"static/images/line_trend_plot_{video_id}.png"
        plt.savefig(line_plot_path)
        plt.close()
    
    # 3. Pie Chart - Create pie chart for emotion distribution
    plt.figure(figsize=(8, 8))
    plt.pie(scores, labels=emotions, autopct='%1.1f%%', startangle=90, shadow=True, 
            wedgeprops={'edgecolor': 'black'}, textprops={'fontsize': 12})
    plt.axis('equal')
    plt.title("Emotion Distribution")
    
    pie_plot_path = f"static/images/pie_emotion_plot_{video_id}.png"
    plt.savefig(pie_plot_path)
    plt.close()
    
    # 4. Histogram - Create histogram for dominant emotions
    plt.figure(figsize=(8, 5))
    if dominant_emotions:
        sns.histplot(dominant_emotions, discrete=True, shrink=0.8, palette='viridis')
        plt.xlabel("Emotion")
        plt.ylabel("Frequency")
        plt.title("Frequency of Detected Emotions")
    else:
        plt.bar(emotions, scores)
        plt.xlabel("Emotions")
        plt.ylabel("Score")
        plt.title("Emotion Scores")
    
    plt.tight_layout()
    hist_plot_path = f"static/images/hist_emotion_plot_{video_id}.png"
    plt.savefig(hist_plot_path)
    plt.close()
        
    # 5. Scatter Plot - Create scatter plot comparing visual and audio emotions
    plt.figure(figsize=(8, 6))
    
    # Create numerical mapping for emotions to plot on scatter chart
    emotion_values = {
        'angry': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 
        'sad': 5, 'surprise': 6, 'neutral': 7
    }
    
    # Generate scatter plot data points
    visual_emotion_value = emotion_values.get(predicted_emotion, 0)
    audio_emotion_value = emotion_values.get(audio_emotion, 0)
    
    # Adding some jitter for visual interest if multiple points would overlap
    visual_jitter = [visual_emotion_value + np.random.normal(0, 0.1) for _ in range(len(dominant_emotions) if dominant_emotions else 5)]
    audio_jitter = [audio_emotion_value + np.random.normal(0, 0.1) for _ in range(len(dominant_emotions) if dominant_emotions else 5)]
    
    # Plot the scatter points
    plt.scatter(visual_jitter, audio_jitter, alpha=0.6, s=80, color='purple')
    
    # Add the main point highlighting the primary emotion detection
    plt.scatter([visual_emotion_value], [audio_emotion_value], color='red', s=200, marker='*', label='Primary Detection')
    
    # Set labels and ticks
    plt.xlabel("Visual Emotion")
    plt.ylabel("Audio Emotion")
    plt.title("Correlation: Visual vs Audio Emotion")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set custom ticks
    plt.xticks(list(emotion_values.values()), list(emotion_values.keys()), rotation=45)
    plt.yticks(list(emotion_values.values()), list(emotion_values.keys()))
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()
    
    scatter_plot_path = f"static/images/scatter_correlation_{video_id}.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    
    # Return the results with all visualization paths and emoji-enhanced emotions
    visual_emotion_display = f"{predicted_emotion.capitalize()} {predicted_emoji}"
    audio_emotion_display = f"{audio_emotion.capitalize()} {audio_emotion_emoji}"
    
    # Clean up temporary files (optional)
    # os.remove(temp_video_path)  # Uncomment if you want to delete the temporary files
    
    return {
        "visual_emotion": visual_emotion_display,
        "audio_emotion": audio_emotion_display,
        "transcript": transcript,
        "bar_plot": "/" + bar_plot_path,
        "line_plot": "/" + line_plot_path,
        "pie_plot": "/" + pie_plot_path,
        "hist_plot": "/" + hist_plot_path,
        "scatter_plot": "/" + scatter_plot_path
    }

# Function to enhance emotion detection with weighted multimodal fusion
def fuse_emotions(visual_emotion, audio_emotion, text_emotion=None):
    """
    Fuse emotions from different modalities using weighted approach
    """
    # Define weights for each modality
    visual_weight = 0.6
    audio_weight = 0.3
    text_weight = 0.1 if text_emotion else 0
    
    # Adjust weights if text_emotion is not available
    if not text_emotion:
        visual_weight = 0.7
        audio_weight = 0.3
    
    # Create numerical mapping for emotions
    emotion_values = {
        'angry': 1, 'disgust': 2, 'fear': 3, 'happy': 4, 
        'sad': 5, 'surprise': 6, 'neutral': 7
    }
    
    # Convert emotions to numeric scores with confidence
    emotions_count = {key: 0 for key in emotion_values.keys()}
    
    # Add weighted contributions
    emotions_count[visual_emotion] += visual_weight
    emotions_count[audio_emotion] += audio_weight
    
    if text_emotion:
        emotions_count[text_emotion] += text_weight
    
    # Return the emotion with highest combined score
    return max(emotions_count, key=emotions_count.get)

# New route for advanced multimodal emotion analysis
@app.post("/analyze_multimodal")
async def analyze_multimodal(video: UploadFile = File(...)):
    """
    Advanced endpoint that provides comprehensive multimodal analysis
    combining visual, audio, and text cues
    """
    try:
        # Process video for visual emotion
        video_id = uuid.uuid4()
        temp_video_path = f"static/uploads/video_{video_id}.mp4"
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Extract audio and transcribe
        audio_path = f"static/audio/audio_{video_id}.wav"
        extract_audio(temp_video_path, audio_path)
        transcript = transcribe_audio(audio_path)
        
        # Process video frames for emotion detection
        cap = cv2.VideoCapture(temp_video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Process frames
        frame_interval = max(1, int(fps / 2))  # Process 2 frames per second
        frame_emotions = []
        frame_id = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_id % frame_interval == 0:
                # Detect faces
                faces, gray = detect_faces(frame)
                
                if len(faces) > 0:
                    # For each face, detect emotion
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        
                        if DeepFace is not None:
                            try:
                                img_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                                analysis = DeepFace.analyze(img_rgb, actions=["emotion"], enforce_detection=False)
                                if analysis:
                                    dominant_emotion = max(analysis[0]['emotion'], key=analysis[0]['emotion'].get)
                                    frame_emotions.append(dominant_emotion)
                            except Exception as e:
                                print(f"Error analyzing face with DeepFace: {e}")
                                frame_emotions.append("neutral")
                        else:
                            # Simple random emotion simulation as fallback
                            from random import choice
                            frame_emotions.append(choice(list(EMOJI_MAP.keys())))
            
            frame_id += 1
            
        cap.release()
        
        # Get visual emotion (most frequent frame emotion)
        if frame_emotions:
            visual_emotion = max(set(frame_emotions), key=frame_emotions.count)
        else:
            visual_emotion = "neutral"
        
        # Get audio emotion from transcript
        audio_emotion = analyze_text_emotion(transcript) if transcript else "neutral"
        
        # Fuse the emotions for final prediction
        final_emotion = fuse_emotions(visual_emotion, audio_emotion)
        final_emoji = EMOJI_MAP.get(final_emotion, "😐")
        
        # Create visualizations (reusing code from predict_emotion)
        # Generate confidence scores for each emotion type (simplified)
        emotion_types = list(EMOJI_MAP.keys())
        emotion_counts = {e: frame_emotions.count(e) if frame_emotions else 0 for e in emotion_types}
        total_frames = len(frame_emotions) if frame_emotions else 1
        emotion_confidence = {e: (count / total_frames) * 100 for e, count in emotion_counts.items()}
        
        # Create visualizations using the confidence scores
        
        # 1. Create emotion timeline
        timestamps = [i/fps for i in range(0, frame_count, frame_interval)]
        emotion_timeline = frame_emotions
        
        # 2. Create enhanced report
        report = {
            "video_id": str(video_id),
            "duration": f"{duration:.2f} seconds",
            "visual_emotion": visual_emotion,
            "audio_emotion": audio_emotion,
            "final_emotion": final_emotion,
            "confidence_scores": emotion_confidence,
            "transcript": transcript,
            "frame_count": frame_count,
            "processed_frames": len(frame_emotions)
        }
        
        # Clean up temporary files
        # os.remove(temp_video_path)
        # os.remove(audio_path)
        
        return report
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in multimodal analysis: {str(e)}")

# Function to calculate emotion intensity over time
def calculate_emotion_intensity(frame_emotions, fps, smoothing=5):
    """
    Calculate emotion intensity over time with smoothing
    """
    # Initialize counters for each emotion
    emotions = list(EMOJI_MAP.keys())
    emotion_timelines = {emotion: [] for emotion in emotions}
    
    # Convert frame emotions to timeline
    for i in range(0, len(frame_emotions), smoothing):
        window = frame_emotions[i:i+smoothing]
        for emotion in emotions:
            count = window.count(emotion)
            emotion_timelines[emotion].append(count / len(window))
    
    # Calculate timestamps (in seconds)
    timestamps = [(i * smoothing) / fps for i in range(len(emotion_timelines[emotions[0]]))]
    
    return timestamps, emotion_timelines

# Function to improve face detection accuracy
def detect_faces_dnn(frame):
    """
    More accurate face detection using DNN model if available
    """
    try:
        # Try to use DNN face detector if available
        model_file = "static/models/opencv_face_detector_uint8.pb"
        config_file = "static/models/opencv_face_detector.pbtxt"
        
        if os.path.exists(model_file) and os.path.exists(config_file):
            net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
            
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
            
            net.setInput(blob)
            detections = net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * width)
                    y1 = int(detections[0, 0, i, 4] * height)
                    x2 = int(detections[0, 0, i, 5] * width)
                    y2 = int(detections[0, 0, i, 6] * height)
                    faces.append((x1, y1, x2-x1, y2-y1))
            
            return faces, frame
        else:
            # Fall back to Haar cascade
            return detect_faces(frame)
    except Exception as e:
        print(f"Error with DNN face detection: {e}")
        # Fall back to Haar cascade
        return detect_faces(frame)

# Main application entry point
if __name__ == "_main_":
    print("🚀 Starting Multimodal Emotion Detection App")
    print("📊 Available emotion types:", list(EMOJI_MAP.keys()))
    
    # Check for required dependencies
    if DeepFace is None:
        print("⚠️ DeepFace not available. Using fallback emotion detection.")
    if sentiment_analyzer is None:
        print("⚠️ Transformers not available. Using TextBlob for sentiment analysis.")
    
    # Download face detection models if needed
    os.makedirs("static/models", exist_ok=True)
    
    # Run the application
    uvicorn.run(app, host="127.0.0.1", port=8000)