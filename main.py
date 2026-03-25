from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import requests
import google.generativeai as genai
import stripe
import os
import tempfile
import json
import time
import uuid
import cv2
import mediapipe as mp
import numpy as np
from dotenv import load_dotenv

# Import the new CV module for split-screen
from cv_modules.swing_plane import generate_split_screen

# Load environment variables
load_dotenv()

# Configure API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
stripe.api_key = STRIPE_SECRET_KEY

app = FastAPI(title="Savage Golf API", version="1.0.0")

# CORS for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def health_check():
    return {"status": "Savage Golf Backend is LIVE. Ready to roast."}

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def process_skeleton(video_path):
    """
    Runs MediaPipe Pose detection over the video to draw a biometric skeleton
    on every frame. Returns the path to the newly rendered video, along with 
    a dictionary of timestamps for the detected mistakes.
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("OpenCV could not open video file.")

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # H.264/MPEG-4 strictly requires EVEN dimensions
    if width % 2 != 0: width -= 1
    if height % 2 != 0: height -= 1
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    
    # 🏎️ PERFORMANCE BOOST: Limit max processing FPS to 15.
    # Analyzing 60fps videos takes 4x as long. 15fps is plenty for AI and MediaPipe.
    process_fps = min(fps, 15)
    frame_skip_ratio = int(fps / process_fps)

    output_path = video_path.replace('.mp4', '_skeleton.mp4')
    
    # Use standard avc1 (H.264) codec so iOS natively plays it without fighting us
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, process_fps, (width, height))

    detected_errors = {
        "chicken_wing_ms": [],
        "posture_loss_ms": []
    }
    
    # NEW: Hand Path Tracking for Swing Plane Visualizer
    hand_path_history = []
    
    # Biometric Reference Lines
    initial_butt_x = None
    initial_head_y = None

    frame_count = 0
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 🏎️ PERFORMANCE BOOST: Skip frames to achieve process_fps (15fps instead of 60fps)
            if frame_count % frame_skip_ratio != 0:
                continue
                
            current_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Recolor to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            # Recolor back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 102), thickness=4, circle_radius=4), # Neon Green
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2)  # White joints
                )
                
                # Biometric Error Tracking Math
                try:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Track Lead Arm (Left Arm for Right-Handed Golfer)
                    l_sh = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    l_el = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    l_wr = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    arm_angle = calculate_angle(l_sh, l_el, l_wr)
                    
                    # Convert to pixel coordinates for OpenCV drawing
                    h, w, _ = image.shape
                    shoulder_px = tuple(np.multiply(l_sh, [w, h]).astype(int))
                    elbow_px = tuple(np.multiply(l_el, [w, h]).astype(int))
                    wrist_px = tuple(np.multiply(l_wr, [w, h]).astype(int))
                    
                    # Early Extension tracking (Hips moving towards the ball too early)
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    
                    spine_angle = calculate_angle(l_sh, l_hip, l_knee)
                    
                    # Tush Line & Head Box Logic
                    nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    
                    if initial_butt_x is None or initial_head_y is None:
                        # Dynamic sizing based on the golfer's body proportions, not the frame size!
                        # Calculate torso length as a reference unit
                        torso_len_y = abs(l_sh[1] - l_hip[1])
                        
                        # Head offset: Top of head is roughly 30% of torso length above the nose
                        head_offset = torso_len_y * 0.3 * h
                        
                        # Glute offset: Back of glutes is roughly 30% of torso length behind the hip joint
                        glute_offset = torso_len_y * 0.3 * w
                        
                        if initial_butt_x is None:
                            hands_x = l_wrist[0]
                            hips_x = (l_hip[0] + r_hip[0]) / 2.0
                            
                            if hands_x > hips_x:
                                # Facing right. Butt is on the left.
                                butt_x = min(l_hip[0], r_hip[0]) * w - glute_offset
                            else:
                                # Facing left. Butt is on the right.
                                butt_x = max(l_hip[0], r_hip[0]) * w + glute_offset
                                
                            initial_butt_x = int(butt_x)
                            
                        if initial_head_y is None:
                            head_top_y = (nose[1] * h) - head_offset
                            initial_head_y = int(head_top_y)
                        
                    # Draw Tush Line (Vertical line at initial butt position)
                    cv2.line(image, (initial_butt_x, 0), (initial_butt_x, h), (0, 255, 255), 2) # Yellow line
                    cv2.putText(image, "TUSH LINE", (initial_butt_x - 100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(image, "TUSH LINE", (initial_butt_x - 100, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                    # Draw Head Height Line (Horizontal line at initial head position)
                    cv2.line(image, (0, initial_head_y - 20), (w, initial_head_y - 20), (255, 165, 0), 2) # Orange line
                    cv2.putText(image, "HEAD LEVEL", (50, initial_head_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    cv2.putText(image, "HEAD LEVEL", (50, initial_head_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)

                    # 1. Chicken Wing Check (Lead arm bending too much)
                    if arm_angle < 135: 
                        # Sleek Red Reticle on the joint
                        cv2.circle(image, elbow_px, 6, (0, 0, 255), -1) # Small Red Center
                        cv2.circle(image, elbow_px, 14, (0, 0, 255), 2) # Red Outer Ring
                        cv2.line(image, shoulder_px, elbow_px, (0, 0, 255), 4) # Thinner, sleeker bone line
                        cv2.line(image, elbow_px, wrist_px, (0, 0, 255), 4)
                        
                        # Burn the text right next to the elbow (with a black drop shadow for readability)
                        text_pos = (elbow_px[0] + 25, elbow_px[1] + 5)
                        cv2.putText(image, "CHICKEN WING", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(image, "CHICKEN WING", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
                                   
                    # 2. Early Extension / Posture Loss Check (Spine standing up too straight)
                    if spine_angle > 170:
                        hip_px = tuple(np.multiply(l_hip, [w, h]).astype(int))
                        
                        # Sleek Red Reticle on the hip
                        cv2.circle(image, hip_px, 6, (0, 0, 255), -1)
                        cv2.circle(image, hip_px, 14, (0, 0, 255), 2)
                        
                        # Burn the text right next to the hip
                        text_pos2 = (hip_px[0] + 25, hip_px[1] + 5)
                        cv2.putText(image, "POSTURE LOSS", text_pos2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(image, "POSTURE LOSS", text_pos2, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

                except Exception as e:
                    pass
            
            # Ensure frame size strictly matches VideoWriter dims
            if image.shape[0] != height or image.shape[1] != width:
                image = cv2.resize(image, (width, height))
                
            out.write(image)

    cap.release()
    out.release()
    return output_path

@app.post("/api/analyze-swing")
async def analyze_swing(video: UploadFile = File(...)):
    """
    Receives a golf swing video from the app, sends it to Gemini 1.5 Pro/Flash
    for biomechanical analysis and a savage roast, and returns JSON.
    """
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API Key missing")

    try:
        import time
        
        # Create a unique ID for this video
        file_id = str(uuid.uuid4())
        temp_video_path = f"static/{file_id}.mp4"
        
        # 1. Save uploaded video to our static folder
        content = await video.read()
        with open(temp_video_path, "wb") as f:
            f.write(content)

        # 1.5 Draw Biometric Skeleton via MediaPipe
        print("Rendering biometric skeleton overlay...")
        skeleton_video_path = process_skeleton(temp_video_path)

        # 2. Upload to Gemini File API
        print(f"Uploading skeleton video to Gemini...")
        gemini_file = genai.upload_file(path=skeleton_video_path)
        
        # Wait for the file to be processed
        print(f"Waiting for {gemini_file.name} to be processed...")
        while gemini_file.state.name == "PROCESSING":
            time.sleep(2)
            gemini_file = genai.get_file(gemini_file.name)
            
        if gemini_file.state.name == "FAILED":
            raise ValueError("Video processing failed in Gemini.")

        
        # 3. Prompt Gemini (Structured JSON output)
        prompt = """
        You are an elite PGA Tour biomechanics coach and a witty, tough-love golf critic. Watch this swing closely.
        First, visually identify the golf club being used (Driver vs Iron vs Wedge vs Putter). Do not hallucinate a driver if they are holding an iron.
        
        Output ONLY valid JSON containing:
        1. 'detected_club': The name of the club you visually identified (e.g., "7-Iron", "Driver", "Wedge").
        2. 'step_by_step_analysis': array of 3-4 strings detailing the breakdown.
        3. 'swing_summary': object with 'posture_score' (1-10), 'tempo' (Fast/Smooth/Jerky), 'estimated_outcome' (Slice/Hook/Pure), 'swing_plane' (Over Top/Under/On Plane), 'clubface_angle' (Open/Closed/Square), 'hip_depth' (Maintained/Loss/Thrust).
        4. 'the_good': one thing they did well.
        5. 'the_critical_flaw': the biggest issue.
        6. 'personalized_training_plan': array of 2 objects. Each must have 'drill_name', 'location' (Driving Range / Living Room), 'how_to_do_it' (2 quick steps), and 'what_to_feel' (a highly specific, exaggerated physical sensation they must focus on during the drill).
        7. 'savage_mode': A 2-sentence verdict. Sentence 1: A witty, punchy roast of their swing. Sentence 2: A clear, educational explanation of exactly what they did wrong biomechanically so they actually learn how to fix it.
        CRITICAL: Never use double quotes (") inside your string values, use single quotes (') instead so you do not break the JSON format.        
        """
                                           
        # Using gemini-3-flash-preview for video capabilities
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(
            [prompt, gemini_file],
            generation_config={"response_mime_type": "application/json"}
        )

        # 4. Clean up the original file & Gemini file
        os.remove(temp_video_path)
        genai.delete_file(gemini_file.name)
        # We DO NOT delete final_video_path because we want to serve it to the app!

        # 5. Parse and return the JSON
        raw_text = response.text
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json\n", "", 1)
        if raw_text.endswith("```\n"):
            raw_text = raw_text[:-4]
        elif raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        
        try:
            analysis_data = json.loads(raw_text.strip())
        except Exception:
            import re
            match = re.search(r'\{.*\}', raw_text.strip(), re.DOTALL)
            analysis_data = json.loads(match.group(0))
        
        # If Gemini returned a list of 1 object instead of an object, extract it
        if isinstance(analysis_data, list):
            analysis_data = analysis_data[0]
            
        # Add the URL for the skeleton video to the response so the iPhone app can stream it!
        analysis_data["skeleton_video_url"] = f"/{skeleton_video_path}"
        
        return analysis_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Handles Stripe webhooks (e.g., when a user subscribes or trial ends).
    """
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        print(f"Payment successful for session: {session['id']}")
        # TODO: Update user in database as 'Premium'
    
    return {"status": "success"}

# TODO: Add ElevenLabs endpoint for generating audio roasts

class AudioRequest(BaseModel):
    text: str
    # Using a default savage-sounding voice ID from ElevenLabs, e.g., "pNInz6obbf5AWCG1MDp1" (Adam)
    voice_id: str = "pNInz6obbf5AWCG1MDp1" 

@app.post("/api/generate-roast-audio")
async def generate_roast_audio(req: AudioRequest):
    """
    Takes the roast text generated by Gemini and turns it into a savage voice memo
    using ElevenLabs TTS.
    """
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API Key missing")
        
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{req.voice_id}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": req.text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    import uuid
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"ElevenLabs Error: {response.text}")
        
    # Save to static folder and return URL
    audio_id = str(uuid.uuid4())
    audio_path = f"static/{audio_id}_roast.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
        
    return {"audio_url": f"/{audio_path}"}
