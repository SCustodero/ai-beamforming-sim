import numpy as np
import matplotlib.pyplot as plt

# Array parameters
num_antennas = 16
wavelength = 1.0
spacing = wavelength / 2 # Half wavelength spacing
antenna_indicies = np.arange(num_antennas)


def steering_vector(angle_rad, spacing, wavelength, antenna_indicies):
    """Computer the steering vector for a uniform linear array at a given angle."""
    phase = 2 * np.pi * spacing / wavelength * antenna_indicies * np.sin(angle_rad)
    return np.exp(1j * phase)


def beam_pattern(target_angle_rad, spacing, wavelength, antenna_indicies, test_angles_rad):
    """Compute gain across all test angles for a beam steered at target_angle_rad."""
    weights = steering_vector(target_angle_rad, spacing, wavelength, antenna_indicies)
    gains = []
    for test_angle in test_angles_rad:
        a_test = steering_vector(test_angle, spacing, wavelength, antenna_indicies)
        gain = np.abs(np.sum(weights * np.conj(a_test)))
        gains.append(gain)
    return np.array(gains)


# Steer towards 30 degrees
target_angle_deg = 30
target_angle_rad = np.radians(target_angle_deg)

test_angles_deg = np.linspace(-90, 90, 500) 
test_angles_rad = np.radians(test_angles_deg)

gains = beam_pattern(target_angle_rad, spacing, wavelength, antenna_indicies, test_angles_rad)
gains_normalized = gains / np.max(gains) # normalize to 0-1
gains_db = 20 * np.log10(gains_normalized + 1e-12) # convert to dB which is the standard for beamforming

# Plot as a polar beam pattern
plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)
ax.plot(test_angles_rad, np.clip(gains_db, -40, 0)) # clip for readable plot
ax.set_theta_zero_location("N") # 0 degrees points "up" (like broadside)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)
ax.set_title(f"Beam Pattern - Steered to {target_angle_deg}°, {num_antennas} antennas")
plt.savefig("phase1_beampattern.png")
plt.show()

print(f"Steering vector at {target_angle_deg}°:")
print(steering_vector(target_angle_rad, spacing, wavelength, antenna_indicies))