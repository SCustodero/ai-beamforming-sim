import numpy as np
import matplotlib.pyplot as plt
import random

# Generate a basic sine wave to represent a signal
frequency = 5  # Hz
t = np.linspace(0, 1, 500)  # 500 points over 1 second
signal = np.sin(2 * np.pi * frequency * t)

magnitude = 1

# Apply a phase shift which is how the signals combine to form a beam and cancel out where there should not be a beam
phase_shift = np.pi / 4 # 45 degrees in radians
shifted_signal = magnitude * np.sin(2 * np.pi * frequency * t + phase_shift)

# Represent the shift as a complex number which is how beamforming weights are represented, ie magnitude + phase
weight = magnitude * (np.exp(1j * phase_shift)) # magnitude 1, phase = 45 degrees
print(f"Complex weight: {weight}")
print(f"Magnitude: {np.abs(weight):.3f}")
print(f"Phase (radians): {np.angle(weight):.3f}")
print(f"Phase (degrees): {np.degrees(np.angle(weight)):.3f}")

# Plot both signals
plt.figure(figsize=(10,4))
plt.plot(t, signal, label="Original signal")
plt.plot(t, shifted_signal, label="Phase-shifted signal", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Signal vs. Phase-Shifted Signal")
plt.legend()
plt.grid(True)
plt.savefig("phase0_output.png")
plt.show()