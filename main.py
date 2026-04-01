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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    Runs MediaPipe Pose detection over the video and burns in:
    - full-body skeleton
    - head level line
    - tush line
    - diagnostic labels

    Returns:
      (output_path, overlay_found)
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("OpenCV could not open video file.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 720
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1280
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1 or fps > 120:
        fps = 30

    process_fps = min(float(fps), 10.0)
    frame_skip_ratio = max(1, round(float(fps) / process_fps))

    output_path = video_path.replace(".mp4", "_skeleton.mp4")

    # More reliable in hosted environments than avc1
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("ERROR: VideoWriter failed to open")
        return None

    overlay_found = False
    pose_frames_detected = 0
    frame_count = 0

    # Stable reference lines
    initial_butt_x = None
    initial_head_y = None
    stance_direction = None # "left" or "right"

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45
    ) as pose:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % frame_skip_ratio != 0:
                continue

            # Resize protection in case source metadata is weird
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            # Back to BGR for OpenCV
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Always stamp overlay watermark so we know processed video is playing
            cv2.putText(
                image,
                "BREAKING 90 AI OVERLAY",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 0),
                5,
                cv2.LINE_AA
            )
            cv2.putText(
                image,
                "BREAKING 90 AI OVERLAY",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 102),
                2,
                cv2.LINE_AA
            )

            if results.pose_landmarks:
                pose_frames_detected += 1
                overlay_found = True

                try:
                    landmarks = results.pose_landmarks.landmark
                    h, w, _ = image.shape

                    # Draw bold premium skeleton
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 102), thickness=6, circle_radius=5),
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=3)
                    )

                    # Key landmarks
                    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                    l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                    r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                    l_el = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                    l_wr = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
                    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                    # Convert to pixel coordinates
                    l_sh_px = tuple(np.multiply([l_sh.x, l_sh.y], [w, h]).astype(int))
                    l_el_px = tuple(np.multiply([l_el.x, l_el.y], [w, h]).astype(int))
                    l_wr_px = tuple(np.multiply([l_wr.x, l_wr.y], [w, h]).astype(int))
                    l_hip_px = tuple(np.multiply([l_hip.x, l_hip.y], [w, h]).astype(int))
                    r_hip_px = tuple(np.multiply([r_hip.x, r_hip.y], [w, h]).astype(int))
                    nose_px = tuple(np.multiply([nose.x, nose.y], [w, h]).astype(int))

                    # Angles
                    arm_angle = calculate_angle(
                        [l_sh.x, l_sh.y],
                        [l_el.x, l_el.y],
                        [l_wr.x, l_wr.y]
                    )

                    spine_angle = calculate_angle(
                        [l_sh.x, l_sh.y],
                        [l_hip.x, l_hip.y],
                        [l_knee.x, l_knee.y]
                    )

                    # Determine golfer direction once from hand/hip relationship
                    if stance_direction is None:
                        hands_x = l_wr.x
                        hips_x = (l_hip.x + r_hip.x) / 2.0
                        stance_direction = "right" if hands_x > hips_x else "left"

                    # Stable head/tush reference setup
                    if initial_butt_x is None or initial_head_y is None:
                        torso_len = abs(l_sh.y - l_hip.y)
                        torso_len = max(torso_len, 0.08)

                        head_offset_px = int(torso_len * h * 0.38)
                        glute_offset_px = int(torso_len * w * 0.32)

                        if initial_butt_x is None:
                            if stance_direction == "right":
                                butt_x = min(l_hip.x, r_hip.x) * w - glute_offset_px
                            else:
                                butt_x = max(l_hip.x, r_hip.x) * w + glute_offset_px
                            initial_butt_x = int(butt_x)

                        if initial_head_y is None:
                            initial_head_y = int((nose.y * h) - head_offset_px)

                    # Draw head level line
                    if initial_head_y is not None:
                        head_line_y = max(20, min(h - 20, initial_head_y))
                        cv2.line(image, (0, head_line_y), (w, head_line_y), (0, 140, 255), 4)
                        cv2.putText(
                            image,
                            "HEAD LINE",
                            (40, max(40, head_line_y - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 0),
                            4,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            image,
                            "HEAD LINE",
                            (40, max(40, head_line_y - 15)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 140, 255),
                            2,
                            cv2.LINE_AA
                        )

                    # Draw tush line
                    if initial_butt_x is not None:
                        butt_line_x = max(20, min(w - 20, initial_butt_x))
                        cv2.line(image, (butt_line_x, 0), (butt_line_x, h), (0, 255, 255), 4)
                        label_x = butt_line_x - 160 if butt_line_x > w // 2 else butt_line_x + 20
                        label_x = max(20, min(w - 220, label_x))

                        cv2.putText(
                            image,
                            "TUSH LINE",
                            (label_x, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 0),
                            4,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            image,
                            "TUSH LINE",
                            (label_x, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                    # Chicken wing visual
                    if arm_angle < 145:
                        cv2.circle(image, l_el_px, 10, (0, 0, 255), -1)
                        cv2.circle(image, l_el_px, 22, (0, 0, 255), 3)
                        cv2.line(image, l_sh_px, l_el_px, (0, 0, 255), 6)
                        cv2.line(image, l_el_px, l_wr_px, (0, 0, 255), 6)

                        text_pos = (l_el_px[0] + 25, l_el_px[1] - 10)
                        cv2.putText(
                            image,
                            "CHICKEN WING",
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (0, 0, 0),
                            4,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            image,
                            "CHICKEN WING",
                            text_pos,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA
                        )

                    # Posture loss visual
                    if spine_angle > 168:
                        hip_mid_px = (
                            int((l_hip_px[0] + r_hip_px[0]) / 2),
                            int((l_hip_px[1] + r_hip_px[1]) / 2)
                        )

                        cv2.circle(image, hip_mid_px, 10, (255, 80, 80), -1)
                        cv2.circle(image, hip_mid_px, 22, (255, 80, 80), 3)

                        text_pos2 = (hip_mid_px[0] + 25, hip_mid_px[1] - 10)
                        cv2.putText(
                            image,
                            "POSTURE LOSS",
                            text_pos2,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (0, 0, 0),
                            4,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            image,
                            "POSTURE LOSS",
                            text_pos2,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.85,
                            (255, 80, 80),
                            2,
                            cv2.LINE_AA
                        )

                    # Nose marker for head tracking visibility
                    cv2.circle(image, nose_px, 8, (0, 140, 255), -1)

                except Exception as draw_error:
                    print(f"Overlay drawing error: {draw_error}")

            out.write(image)

    cap.release()
    out.release()

    print(f"Pose frames detected: {pose_frames_detected}")

    if not overlay_found:
        print("No pose landmarks detected in any frame.")
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except Exception as cleanup_error:
            print(f"Could not remove empty overlay video: {cleanup_error}")
        return None, False

    if not os.path.exists(output_path) or os.path.getsize(output_path) <= 1024:
        return None, False

    return output_path, True
    
def create_analysis_clip(video_path, max_seconds=4, target_height=540, target_fps=8):
    """
    Creates a shorter, smaller MP4 clip for faster Gemini analysis.
    Returns the path to the compressed analysis clip.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Could not open video for analysis clip creation.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 720
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1280

    scale = target_height / float(height)
    target_width = int(width * scale)
    target_width = max(2, target_width - (target_width % 2))
    target_height = max(2, target_height - (target_height % 2))

    output_path = video_path.replace(".mp4", "_analysis.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height))

    if not out.isOpened():
        raise Exception("Could not open VideoWriter for analysis clip.")

    max_frames = int(max_seconds * fps)
    frame_skip_ratio = max(1, round(fps / target_fps))

    frame_count = 0
    written_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count > max_frames:
            break

        if frame_count % frame_skip_ratio != 0:
            continue

        resized = cv2.resize(frame, (target_width, target_height))
        out.write(resized)
        written_count += 1

    cap.release()
    out.release()

    if written_count == 0 or not os.path.exists(output_path) or os.path.getsize(output_path) <= 1024:
        raise Exception("Analysis clip creation failed.")

    return output_path

def render_swing_overlay_video(input_video_path: str, file_id: str):
    import os
    import shutil

    print("=== OVERLAY START ===")
    print(f"input_video_path={input_video_path}")

    if not os.path.exists(input_video_path):
        print("Overlay input video does not exist")
        return None

    output_path = f"static/{file_id}_overlay.mp4"
    print(f"output_path={output_path}")

    try:
        shutil.copy(input_video_path, output_path)
        print(f"Overlay test copy created at {output_path}")
        return output_path
    except Exception as e:
        print(f"Overlay copy failed: {repr(e)}")
        return None
def render_swing_overlay_video(input_video_path: str, file_id: str):
    import os
    import cv2
    import mediapipe as mp

    print("=== OVERLAY START ===")
    print(f"input_video_path={input_video_path}")

    if not os.path.exists(input_video_path):
        print("Overlay input video does not exist")
        return None

    output_path = f"static/{file_id}_overlay.mp4"
    print(f"output_path={output_path}")

    cap = None
    out = None

    try:
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print("ERROR: Could not open input video")
            return None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0 or fps > 120:
            fps = 24.0

        # Cap output FPS for faster processing / smaller files
        fps = min(fps, 15.0)

        print(f"video info: width={width}, height={height}, fps={fps}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            print("ERROR: VideoWriter failed to open")
            return None

        mp_pose = mp.solutions.pose

        frame_count = 0
        pose_frames = 0

        head_line_y = None
        tush_line_x = None
        setup_locked = False

        last_nose_x = None
        last_nose_y = None
        last_hip_x = None
        last_hip_y = None

        last_results = None

        NOSE = mp_pose.PoseLandmark.NOSE
        RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                output_frame = output_frame.copy()
                out.write(output_frame)

                # Run pose only every 2nd frame for speed
                if frame_count % 2 == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_results = pose.process(rgb)

                results = last_results

                if results and results.pose_landmarks:
                    pose_frames += 1
                    landmarks = results.pose_landmarks.landmark

                    nose_x = int(landmarks[NOSE].x * width)
                    nose_y = int(landmarks[NOSE].y * height)

                    hip_x = int(landmarks[RIGHT_HIP].x * width)
                    hip_y = int(landmarks[RIGHT_HIP].y * height)

                    last_nose_x = nose_x
                    last_nose_y = nose_y
                    last_hip_x = hip_x
                    last_hip_y = hip_y

                    if not setup_locked:
                        head_line_y = nose_y
                        tush_line_x = hip_x
                        setup_locked = True
                        print(f"Setup locked: head_line_y={head_line_y}, tush_line_x={tush_line_x}")

                # Draw HEAD LINE
                if head_line_y is not None:
                    cv2.line(output_frame, (0, head_line_y), (width, head_line_y), (0, 0, 0), 6)
                    cv2.line(output_frame, (0, head_line_y), (width, head_line_y), (255, 255, 0), 3)

                    cv2.putText(
                        output_frame,
                        "HEAD LINE",
                        (20, max(35, head_line_y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        output_frame,
                        "HEAD LINE",
                        (20, max(35, head_line_y - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                # Draw TUSH LINE
                if tush_line_x is not None:
                    cv2.line(output_frame, (tush_line_x, 0), (tush_line_x, height), (0, 0, 0), 6)
                    cv2.line(output_frame, (tush_line_x, 0), (tush_line_x, height), (0, 140, 255), 3)

                    label_x = min(tush_line_x + 12, width - 180)

                    cv2.putText(
                        output_frame,
                        "TUSH LINE",
                        (label_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 0),
                        4,
                        cv2.LINE_AA
                    )
                    cv2.putText(
                        output_frame,
                        "TUSH LINE",
                        (label_x, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 140, 255),
                        2,
                        cv2.LINE_AA
                    )

                # Draw current markers
                if last_nose_x is not None and last_nose_y is not None:
                    cv2.circle(output_frame, (last_nose_x, last_nose_y), 8, (0, 0, 0), -1)
                    cv2.circle(output_frame, (last_nose_x, last_nose_y), 5, (255, 255, 0), -1)

                if last_hip_x is not None and last_hip_y is not None:
                    cv2.circle(output_frame, (last_hip_x, last_hip_y), 8, (0, 0, 0), -1)
                    cv2.circle(output_frame, (last_hip_x, last_hip_y), 5, (0, 140, 255), -1)

                out.write(output_frame)

        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

        exists = os.path.exists(output_path)
        size = os.path.getsize(output_path) if exists else 0

        print(f"processed frames={frame_count}, pose_frames={pose_frames}")
        print(f"output exists={exists}, size={size}")

        if exists and size > 0 and pose_frames > 0:
            print("=== OVERLAY SUCCESS ===")
            return output_path

        print("ERROR: Overlay output missing, empty, or no pose frames detected")
        return None

    except Exception as e:
        print(f"OVERLAY EXCEPTION: {repr(e)}")
        return None

    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        try:
            if out is not None:
                out.release()
        except Exception:
            pass
            
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
        import os
        analysis_video_path = None
        
        # CRITICAL FIX: Ensure static directory exists
        if not os.path.exists("static"):
            os.makedirs("static", exist_ok=True)
            
        # Create a unique ID for this video
        file_id = str(uuid.uuid4())
        temp_video_path = f"static/{file_id}.mp4"
        
        # 1. Save uploaded video to our static folder
        content = await video.read()
        with open(temp_video_path, "wb") as f:
            f.write(content)
        # 1.2 Create a shorter compressed clip for faster AI analysis
        print("Creating short analysis clip...")
        analysis_video_path = create_analysis_clip(temp_video_path, max_seconds=4, target_height=540, target_fps=8)
        print("Rendering skeleton + head/tush overlay video...")
        try:
            skeleton_video_path = render_swing_overlay_video(analysis_video_path, file_id=file_id)
            print(f"skeleton_video_path returned: {skeleton_video_path}")
            use_processed_video = True if skeleton_video_path else False

            if not skeleton_video_path:
                print("Overlay renderer returned None, falling back to original video.")
                skeleton_video_path = temp_video_path
        except Exception as overlay_error:
            print(f"Overlay rendering failed: {repr(overlay_error)}")
            skeleton_video_path = temp_video_path
            use_processed_video = False
            
        # 2. Upload ORIGINAL video to Gemini for faster analysis
        print("Uploading compressed analysis clip to Gemini...")
        gemini_file = genai.upload_file(path=analysis_video_path)

        # Wait for the file to be processed, but do not hang forever
        print(f"Waiting for {gemini_file.name} to be processed...")
        max_wait_seconds = 45
        waited = 0

        while gemini_file.state.name == "PROCESSING" and waited < max_wait_seconds:
            time.sleep(2)
            waited += 2
            gemini_file = genai.get_file(gemini_file.name)

        if gemini_file.state.name == "PROCESSING":
            raise ValueError("Gemini processing timed out.")

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
        8. 'fitness_prescription': array of 2-3 objects to fix their physical limitations. Each must have 'exercise_name', 'sets_and_reps', and 'why_it_helps'.
        9. 'physical_diagnosis': A witty, brutal explanation of the physical limitation in their body that caused the swing flaw (e.g. tight hips, weak core).
        
        CRITICAL: Never use double quotes (") inside your string values, use single quotes (') instead so you do not break the JSON format.
        """

        # Using gemini-3-flash-preview for video capabilities
        model = genai.GenerativeModel('models/gemini-3-flash-preview')
        response = model.generate_content(
            [prompt, gemini_file],
            generation_config={"response_mime_type": "application/json"}
        )

        # 4. Clean up Gemini file and temporary analysis clip
        genai.delete_file(gemini_file.name)

        try:
            if analysis_video_path and os.path.exists(analysis_video_path):
                os.remove(analysis_video_path)
        except Exception as cleanup_error:
            print(f"Warning: could not delete analysis clip: {cleanup_error}")
            
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
        except json.JSONDecodeError as jde:
            # Fallback regex extraction if json loads fails
            import re
            match = re.search(r'\{.*\}', raw_text.strip(), re.DOTALL)
            if match:
                analysis_data = json.loads(match.group(0))
            else:
                raise ValueError(f"JSON Parse Error: {jde}. Raw text: {raw_text[:200]}...")
        
        # If Gemini returned a list of 1 object instead of an object, extract it
        if isinstance(analysis_data, list):
            analysis_data = analysis_data[0]
            
        # Add the URL for the skeleton video to the response so the iPhone app can stream it!
        analysis_data["skeleton_video_url"] = f"/{skeleton_video_path}"
        analysis_data["overlay_available"] = use_processed_video        
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
    # Using 'onyx' from OpenAI
    voice_id: str = "onyx" 

@app.post("/api/generate-roast-audio")
async def generate_roast_audio(req: AudioRequest):
    """
    Takes the roast text generated by Gemini and turns it into a savage voice memo
    using OpenAI TTS.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API Key missing")
        
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "tts-1",
        "input": req.text,
        "voice": req.voice_id
    }
    
    import uuid
    import os
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=f"OpenAI TTS Error: {response.text}")
        
    # CRITICAL FIX: Ensure static directory exists before writing audio
    if not os.path.exists("static"):
        os.makedirs("static", exist_ok=True)
        
    # Save to static folder and return URL
    audio_id = str(uuid.uuid4())
    audio_path = f"static/{audio_id}_roast.mp3"
    with open(audio_path, "wb") as f:
        f.write(response.content)
        
    return {"audio_url": f"/{audio_path}"}
