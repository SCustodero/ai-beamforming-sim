import numpy as np
import matplotlib.pyplot as plt

num_antennas = 8
wavelength = 1.0
spacing = wavelength / 2
antenna_indicies = np.arange(num_antennas)


def steering_vector(angle_rad, spacing, wavelength, antenna_indicies):
    phase = 2 * np.pi * spacing / wavelength * antenna_indicies * np.sin(angle_rad)
    return np.exp(1j * phase)


# Test scenario: desired signal at 30°, interferer at -20°
desired_angle_deg = 30
interferer_angle_deg = -20
noise_power = 0.01

desired_angle_rad = np.radians(desired_angle_deg)
interferer_angle_rad = np.radians(interferer_angle_deg)

a_desired = steering_vector(desired_angle_rad, spacing, wavelength, antenna_indicies)
a_interferer = steering_vector(interferer_angle_rad, spacing, wavelength, antenna_indicies)

# Build covariance matrix R
# R = signal_power * a_d *a_d^H + interferer_power * a_i *a_i^H + noise_power * I
signal_power = 1.0
interferer_power = 5

R = (signal_power * np.outer(a_desired, np.conjugate(a_desired)) + \
    interferer_power * np.outer(a_interferer, np.conj(a_interferer)) + \
    noise_power * np.eye(num_antennas))

# MVDR Weights
R_inv = np.linalg.inv(R)
numerator = R_inv @ a_desired
denominator = np.conj(a_desired) @ R_inv @ a_desired
mvdr_weights = numerator / denominator

# Compare against simple steering (Phase 1 style)
simple_weights = a_desired


def compute_pattern(weights, spacing, wavelength, antenna_indicies, test_angles_rad):
    gains = []
    for test_angle in test_angles_rad:
        a_test = steering_vector(test_angle, spacing, wavelength, antenna_indicies)
        gain = np.abs(np.conj(weights) @ a_test)
        gains.append(gain)
    return np.array(gains)


test_angles_deg = np.linspace(-90, 90, 500)
test_angles_rad = np.radians(test_angles_deg)

mvdr_gains = compute_pattern(mvdr_weights, spacing, wavelength, antenna_indicies, test_angles_rad)
simple_gains = compute_pattern(simple_weights, spacing, wavelength, antenna_indicies, test_angles_rad)

mvdr_db = 20 * np.log10(mvdr_gains / np.max(mvdr_gains) + 1e-12)
simple_db = 20 * np.log10(simple_gains / np.max(simple_gains) + 1e-12)

# Plot linear axis to compare directly
plt.figure(figsize=(10,5))
plt.plot(test_angles_deg, np.clip(simple_db, -40, 0), label="Simple steering", linestyle="--")
plt.plot(test_angles_deg, np.clip(mvdr_db, -40, 0), label="MVDR (adaptive)")
plt.axvline(desired_angle_deg, color="green", linestyle=":", label="Desired signal")
plt.axvline(interferer_angle_deg, color="red", linestyle=":", label="Interferer")
plt.xlabel("Angle (degrees)")
plt.ylabel("Gain (dB)")
plt.title("Simple Steering vs. MVDR Adaptive Beamforming")
plt.legend()
plt.grid(True)
plt.savefig("phase1b_nulling.png")
plt.show()

print(f"Gain toward desired signal (MVDR): {20*np.log10(np.abs(np.conj(mvdr_weights) @ a_desired)):.2f} dB")
print(f"Gain toward interferer (MVDR): {20*np.log10(np.abs(np.conj(mvdr_weights) @ a_interferer)):.2f} dB")
print(f"Gain toward interferer (simple steering): {20*np.log10(np.abs(np.conj(simple_weights) @ a_interferer)):.2f} dB")