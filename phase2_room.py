import numpy as np
import matplotlib.pyplot as plt

num_antennas = 16
wavelength = 2.0
spacing = wavelength / 2
antenna_indicies = np.arange(num_antennas)

def steering_vector(angle_rad, spacing, wavelength, antenna_indicies):
    phase = 2 * np.pi * spacing / wavelength * antenna_indicies * np.sin(angle_rad)
    return np.exp(1j * phase)

# Room setup
# Array is centered at origin, facing "up" (+y direction), broadside at 0 degrees
room_width = 10 # meters, x-axis
room_height = 8 # meters, y-axis (array sits at y=0, room extends upward)

array_position = np.array([room_width / 2, 0.0])

def angle_and_distance(array_pos, point):
    """Angle relative to broadside (+y axis) and distance from array to point."""
    dx = point[0] - array_pos[0]
    dy = point[1] - array_pos[1]
    distance = np.sqrt(dx**2 + dy**2)
    angle_rad = np.arctan2(dx, dy) # angle from the broadside (+y), positive = toward +x
    return angle_rad, distance


def free_space_path_loss(distance, wavelength):
    """Simplified free-space path loss (Friis-like, normalized)."""
    distance = np.maximum(distance, 0.1) # avoid divide-by-zero near the array
    return (wavelength / (4 * np.pi * distance)) ** 2

# Place a receiver and steer towards it
receiver_pos = np.array([2.0, 4.0])
receiver_angle, receiver_distance = angle_and_distance(array_position, receiver_pos)
weights = steering_vector(receiver_angle, spacing, wavelength, antenna_indicies)

print(f"Receiver at {receiver_pos}: angle = {np.degrees(receiver_angle):.1f} deg, distance = {receiver_distance:.2f} m")

# Build a heatmap of received signal strength across the whole room
x_vals = np.linspace(0.1, room_width, 150)
y_vals = np.linspace(0.1, room_height, 150)
signal_map = np.zeros((len(y_vals), len(x_vals)))

for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        point = np.array([x, y])
        angle, dist = angle_and_distance(array_position, point)
        a_point = steering_vector(angle, spacing, wavelength, antenna_indicies)
        array_gain = np.abs(np.conj(weights) @ a_point) ** 2 # beamforming gain at this point
        path_loss = free_space_path_loss(dist, wavelength)
        signal_map[i, j] = array_gain * path_loss

signal_map_db = 10 * np.log10(signal_map / np.max(signal_map) + 1e-12)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(signal_map_db, extent=[0, room_width, 0, room_height], origin="lower",
           cmap="inferno", vmin=-40, vmax=0, aspect="equal")
plt.colorbar(label="Relative signal strength (dB)")
plt.scatter(*array_position, color="cyan", marker="^", s=150, label="Antenna array")
plt.scatter(*receiver_pos, color="lime", marker="o", s=100, label="Receiver")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Room Signal Strength Heatmap - Beam Steered to Receiver")
plt.legend()
plt.savefig("phase2_room_heatmap.png")
plt.show()
