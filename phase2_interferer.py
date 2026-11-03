import numpy as np
import matplotlib.pyplot as plt

# --------- Array and room (same as phase2_room.py) --------------
num_antennas = 16
wavelength = 2.0
spacing = wavelength / 2
antenna_indices = np.arange(num_antennas)

room_width = 10 # meters, x-axis
room_height = 8 # meters, y-axis
array_position = np.array([room_width / 2, 0.0])

wall_x = room_width
reflection_coefficient = 0.6

def steering_vector(angle_rad):
    phase = 2 * np.pi * spacing / wavelength * antenna_indices * np.sin(angle_rad)
    return np.exp(1j * phase)

def angle_and_distance(array_pos, point):
    dx = point[0] - array_pos[0]
    dy = point[1] - array_pos[1]
    return np.arctan2(dx, dy), np.sqrt(dx**2 + dy**2)

def free_space_path_loss(distance):
    distance = np.maximum(distance, 1.0)
    return (wavelength / (4 * np.pi * distance)) ** 2

def mirror_point(point):
    """Image of a point across the wall at x = wall_x."""
    return np.array([2 * wall_x - point[0], point[1]])



# ------ Channel model -------
def arrivals(source_pos):
    """The two ways a signal from the source_pos reaches the array.
        Returns (angle, distance, amplitude_scale) for the direct path and the wall bounce."""
    angle_d, dist_d = angle_and_distance(array_position, source_pos)
    angle_r, dist_r = angle_and_distance(array_position, mirror_point(source_pos))
    return [(angle_d, dist_d, 1.0), (angle_r, dist_r, reflection_coefficient)]

def channel_vector(source_pos):
    """What each antenna hears from a single-antenna transmitter at source_pos
        (direct path  + wall_reflection, including propagation phase)."""
    h = np.zeros(num_antennas, dtype=complex)
    for angle, dist, scale in arrivals(source_pos):
        amplitude = scale * np.sqrt(free_space_path_loss(dist))
        h += amplitude * np.exp(-1j * 2 * np.pi * dist / wavelength) * steering_vector(angle)
    return h

def interference_covariance(source_pos, tx_power, noise_power):
    """Interference-plus-noise covariance seen by the array.
        Each arrival (direct, wall bounce) is treated as its own source from its own direction."""
    R = noise_power * np.eye(num_antennas, dtype=complex)
    for angle, dist, scale in arrivals(source_pos):
        power_at_array = tx_power * scale ** 2 * free_space_path_loss(dist)
        a = steering_vector(angle)
        R += power_at_array * np.outer(a, np.conj(a))
    return R

# ------- Scenario --------
receiver_pos = np.array([8.0, 3.0])
interferer_pos = np.array([8.0, 5.0])

desired_tx_power = 1.0
interferer_tx_power = 1.0 # 30 dB stronger than the desired user (a loud nearby device)
noise_power = 1e-6

receiver_angle, _ = angle_and_distance(array_position, receiver_pos)
interferer_angle, interferer_dist = angle_and_distance(array_position, interferer_pos)

# ------- Three beamformers --------
a_desired = steering_vector(receiver_angle)

w_uniform = a_desired
w_tapered = a_desired * np.hamming(num_antennas)

R = interference_covariance(interferer_pos, interferer_tx_power, noise_power)
Rinv_a = np.linalg.solve(R, a_desired)
w_mvdr = Rinv_a / (np.conj(a_desired) @ Rinv_a)

beamformers = {
    "Uniform steering": w_uniform,
    "Hamming taper": w_tapered,
    "MVDR (adaptive)": w_mvdr,
}
# Unit-norm weights give every beamformer the same noise gain, so the maps are comparable
beamformers = {name: w / np.linalg.norm(w) for name, w in beamformers.items()}

# ------- SINR at the desired user -------
h_desired = channel_vector(receiver_pos)
h_interferer = channel_vector(interferer_pos)

def sinr_db(w):
    signal = desired_tx_power * np.abs(np.conj(w) @ h_desired) ** 2
    interference = interferer_tx_power * np.abs(np.conj(w) @ h_interferer) ** 2
    noise = noise_power * np.linalg.norm(w) ** 2
    return 10 * np.log10(signal / (interference + noise))

snr_element = 10 * np.log10(desired_tx_power * np.mean(np.abs(h_desired) ** 2) / noise_power)
inr_element = 10 * np.log10(interferer_tx_power * np.mean(np.abs(h_interferer) ** 2) / noise_power)

print(f"Desired user: angle = {np.degrees(receiver_angle):.1f} deg")
print(f"Interferer: angle = {np.degrees(interferer_angle):.1f} deg, distance = {interferer_dist:.2f} m")
print(f"Per-antenna SNR (desired): {snr_element:.1f} dB | Per-antenna INR (interferer): {inr_element:.1f} dB")
for name, w in beamformers.items():
                print(f"{name:18s} SINR = {sinr_db(w):6.1f} dB")


# --------- Heatmaps ---------
x_vals = np.linspace(0.1, room_width, 150)
y_vals = np.linspace(0.1, room_height, 150)

# Compute the channel at every grid point once, reuse for all three beamformers
H = np.zeros((len(y_vals), len(x_vals), num_antennas), dtype=complex)
for i, y in enumerate(y_vals):
    for j, x in enumerate(x_vals):
        H[i, j] = channel_vector(np.array([x, y]))

maps = {name: np.abs(H @ np.conj(w)) ** 2 for name, w in beamformers.items()}
reference = max(m.max() for m in maps.values()) # shared reference so panels are comparable

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
for ax, (name, m) in zip(axes, maps.items()):
    m_db = 10 * np.log10(m / reference + 1e-12)
    im = ax.imshow(m_db, extent=[0, room_width, 0, room_height], origin="lower",
                   cmap="inferno", vmin=-40, vmax=0, aspect="equal")
    ax.scatter(*array_position, color="cyan", marker="^", s=150, label="Antenna array")
    ax.scatter(*receiver_pos, color="lime", marker="o", s=100, label="Desired user")
    ax.scatter(*interferer_pos, color="red", marker="X", s=120, label="Interferer")
    ax.axvline(wall_x, color="white", linestyle="--", linewidth=1)
    ax.set_title(f"{name}\nSINR at user: {sinr_db(beamformers[name]):.1f} dB")
    ax.set_xlabel("X (m)")
axes[0].set_ylabel("Y (m)")
axes[0].legend(loc="upper left", fontsize=8)
fig.colorbar(im, ax=axes, label="Array sensitivity (dB)", shrink=0.8)
plt.savefig("phase2_interferer.png", dpi=150)
plt.show()
