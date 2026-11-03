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
room_height = 10 # meters, y-axis (array sits at y=0, room extends upward)

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
    distance = np.maximum(distance, 0.7) # avoid divide-by-zero near the array
    return (wavelength / (4 * np.pi * distance)) ** 2


receiver_pos = np.array([8.0, 3.0])
receiver_angle, receiver_distance = angle_and_distance(array_position, receiver_pos)
weights = steering_vector(receiver_angle, spacing, wavelength, antenna_indicies)

print(f"Receiver at {receiver_pos}: angle = {np.degrees(receiver_angle):.1f} deg, distance = {receiver_distance:.2f} m")

# --- Wall and reflection setup ---
wall_x = room_width # right wall of the room, at x = room_width
reflection_coefficient = 0.6 # fraction of amplitutd that survives the bounce

def field_contribution(source_pos, weights, point, spacing, wavelength, antenna_indicies):
    """Complex field at 'point' from an array at 'source_pos' using given weights.
    Includes array gain and propagation phase (needed for coherent multipath sum)."""
    angle, dist = angle_and_distance(source_pos, point)
    a_point = steering_vector(angle, spacing, wavelength, antenna_indicies)
    array_response = np.conj(weights) @ a_point
    path_loss_amplitude = np.sqrt(free_space_path_loss(dist, wavelength))
    propagation_phase = np.exp(-1j * 2 * np.pi * dist / wavelength)
    return array_response * path_loss_amplitude * propagation_phase

# Build a heatmap of received signal strength across the whole room
x_vals = np.linspace(0.1, room_width, 150)
y_vals = np.linspace(0.1, room_height, 150)
signal_map = np.zeros((len(y_vals), len(x_vals)))

for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        point = np.array([x, y])

        # Direct path: real array -> point
        direct_field = field_contribution(array_position, weights, point,
                                          spacing, wavelength, antenna_indicies)

        # Reflected path: virtual (mirrored) array -> point, attenuated by wall
        mirrored_point = np.array([2 * wall_x - x, y])
        reflected_field = reflection_coefficient * field_contribution(
            array_position, weights, mirrored_point,
            spacing, wavelength, antenna_indicies)

        total_field = direct_field + reflected_field
        signal_map[i, j] = np.abs(total_field) ** 2
        

signal_map_db = 10 * np.log10(signal_map / np.max(signal_map) + 1e-12)

# Plot heatmap
plt.figure(figsize=(8, 6))
plt.imshow(signal_map_db, extent=[0, room_width, 0, room_height], origin="lower",
           cmap="inferno", vmin=-40, vmax=0, aspect="equal")
plt.colorbar(label="Relative signal strength (dB)")
plt.scatter(*array_position, color="cyan", marker="^", s=150, label="Antenna array")
plt.scatter(*receiver_pos, color="lime", marker="o", s=100, label="Receiver")
plt.axvline(wall_x, color="white", linestyle="--", linewidth=1, label="Reflecting wall")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Room Signal Strength Heatmap - Beam Steered to Receiver")
plt.legend()
plt.savefig("phase2_room_heatmap.png")
plt.show()
