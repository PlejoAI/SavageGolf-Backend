import cv2
import numpy as np
import os

def generate_split_screen(user_video_path, output_path, pro_video_path="pro_models/pro_iron.mp4"):
    """
    Takes the user's processed swing video (with skeleton and tracer) and a reference PGA Pro video.
    Synchronizes them and stitches them side-by-side.
    """
    if not os.path.exists(pro_video_path):
        print(f"Warning: Pro model {pro_video_path} not found. Skipping split screen.")
        return user_video_path

    cap_user = cv2.VideoCapture(user_video_path)
    cap_pro = cv2.VideoCapture(pro_video_path)

    if not cap_user.isOpened() or not cap_pro.isOpened():
        print("Error opening video streams for split screen.")
        return user_video_path

    # We will conform the Pro video to the User video's height so they line up nicely.
    u_w = int(cap_user.get(cv2.CAP_PROP_FRAME_WIDTH))
    u_h = int(cap_user.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_user.get(cv2.CAP_PROP_FPS) or 30

    p_w = int(cap_pro.get(cv2.CAP_PROP_FRAME_WIDTH))
    p_h = int(cap_pro.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate new width for the pro video to maintain aspect ratio while matching the user's height
    new_p_w = int(p_w * (u_h / float(p_h)))

    final_width = u_w + new_p_w

    # Ensure dimensions are strictly EVEN numbers (H.264 standard requires this, otherwise iOS shows a black screen)
    if final_width % 2 != 0:
        final_width -= 1
    if u_h % 2 != 0:
        u_h -= 1
        
    # Also ensure the sub-width of the pro video is adjusted so the math adds up perfectly for np.hstack
    new_p_w = final_width - u_w

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (final_width, u_h))

    # A simple frame-by-frame stitch. 
    # (Future optimization: Dynamic Time Warping to sync the exact top of the backswing)
    while True:
        ret_u, frame_u = cap_user.read()
        ret_p, frame_p = cap_pro.read()

        if not ret_u:
            break
            
        # If the pro video is shorter, we just freeze on the last frame of the pro
        if ret_p:
            frame_p_resized = cv2.resize(frame_p, (new_p_w, u_h))
            last_p_frame = frame_p_resized
        else:
            frame_p_resized = last_p_frame if 'last_p_frame' in locals() else np.zeros((u_h, new_p_w, 3), dtype=np.uint8)

        # Force the user frame to match the perfectly even u_h if we adjusted it
        if frame_u.shape[0] != u_h or frame_u.shape[1] != u_w:
            frame_u = cv2.resize(frame_u, (u_w, u_h))

        # Stitch them horizontally (User on Left, Pro on Right)
        split_frame = np.hstack((frame_u, frame_p_resized))
        
        # Add a slick black divider line down the middle
        cv2.line(split_frame, (u_w, 0), (u_w, u_h), (0, 0, 0), 4)
        
        # Add labels
        cv2.putText(split_frame, "YOU", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(split_frame, "YOU", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 102), 2, cv2.LINE_AA)
        
        cv2.putText(split_frame, "PRO", (u_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(split_frame, "PRO", (u_w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(split_frame)

    cap_user.release()
    cap_pro.release()
    out.release()
    
    return output_path
