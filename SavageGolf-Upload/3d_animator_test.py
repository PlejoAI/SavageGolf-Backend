import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- 1. MOCK 3D SWING DATA ---
frames = 60
t = np.linspace(0, 1, frames)

# Pro Swing Path (Neon Green) - Inside to Out, smooth plane
pro_x = np.cos(np.pi * t) * 2
pro_y = np.sin(np.pi * t) * 2
pro_z = np.sin(np.pi * t) * 3

# Flawed Swing Path (Red) - Over The Top (Steep, Outside-In)
flawed_x = np.cos(np.pi * t) * 2.5 - 0.5 * t
flawed_y = np.sin(np.pi * t) * 1.5 + 1.0 * t
flawed_z = np.sin(np.pi * t) * 3.5 + 0.5 * t

# --- 2. SET UP 3D RENDERER ---
fig = plt.figure(figsize=(8, 8), facecolor='#111111')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#111111')

# Hide axes for a clean, premium look
ax.set_axis_off()

# Set to TOP-DOWN view
ax.view_init(elev=75, azim=-90)

# Initialize lines
pro_line, = ax.plot([], [], [], color='#00FF66', lw=4, label='Pro Path (Correct)')
flawed_line, = ax.plot([], [], [], color='#FF3333', lw=4, label='Your Path (Over The Top)')

# Ball position
ax.scatter([0], [2], [0], color='white', s=100)
ax.text(0, 2.2, 0, "Golf Ball", color='white')

# Target line
ax.plot([-3, 3], [2.5, 2.5], [0, 0], color='white', linestyle='--', alpha=0.5)

ax.set_xlim(-3, 3)
ax.set_ylim(-1, 4)
ax.set_zlim(0, 4)
ax.legend(loc='upper right', facecolor='#222222', edgecolor='none', labelcolor='white')

# --- 3. ANIMATION LOGIC ---
def update(num):
    # Draw paths up to the current frame
    pro_line.set_data(pro_x[:num], pro_y[:num])
    pro_line.set_3d_properties(pro_z[:num])
    
    flawed_line.set_data(flawed_x[:num], flawed_y[:num])
    flawed_line.set_3d_properties(flawed_z[:num])
    return pro_line, flawed_line

ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)

# Save as GIF (easier to test locally without FFmpeg installed)
print("Rendering Biomechanical 3D Matrix...")
ani.save('correction_demo.gif', writer='pillow', fps=20)
print("Render Complete! Saved as correction_demo.gif")
