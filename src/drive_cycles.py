"""
Standard Electric Vehicle & Energy Storage Drive Cycles
Journal of Electrochemical Energy Conversion and Storage (ASME)

Generates standardized dynamic current profiles for:
- UDDS (Urban Dynamometer Driving Schedule)
- US06 (High-acceleration aggressive highway cycle)
- DST (Dynamic Stress Test, USABC electric vehicle benchmark)
- FUDS (Federal Urban Driving Schedule)
- WLTP-derived battery dynamic current profiles
"""

import numpy as np


class DriveCycles:
    @staticmethod
    def generate_dst(duration_sec=3600, dt=1.0, current_scale=2.5):
        """
        USABC Dynamic Stress Test (DST) cycle.
        A 360-second power profile repeated for duration_sec.
        Scaled to represent typical EV battery C-rates.
        """
        # 360-second standard DST step profile (step duration in seconds, normalized power/current)
        # Sequence of discharge, regen charge, and rest steps
        steps = [
            (16, 0.4),    # Low discharge
            (28, 0.8),    # Medium discharge
            (12, -0.6),   # High regenerative braking
            (8, 0.0),     # Rest
            (16, 1.2),    # High discharge acceleration
            (24, 0.4),    # Low discharge
            (8, -0.4),    # Regen
            (32, 0.8),    # Medium discharge
            (8, 1.6),     # Peak acceleration
            (12, -0.8),   # Peak regen
            (36, 0.4),    # Cruise discharge
            (160, 0.0),   # Rest / coast
        ]
        
        cycle_pattern = []
        for duration, val in steps:
            num_points = int(duration / dt)
            cycle_pattern.extend([val * current_scale] * num_points)
        
        cycle_len = len(cycle_pattern)
        total_steps = int(duration_sec / dt)
        num_repeats = int(np.ceil(total_steps / cycle_len))
        
        current_profile = np.tile(cycle_pattern, num_repeats)[:total_steps]
        # Add slight realistic stochastic ripple
        np.random.seed(42)
        ripple = np.random.normal(0, 0.02 * current_scale, total_steps)
        return current_profile + ripple

    @staticmethod
    def generate_udds(duration_sec=3600, dt=1.0, max_current_a=5.0):
        """
        UDDS (Urban Dynamometer Driving Schedule) current equivalent profile.
        Simulates city stop-and-go driving with frequent acceleration, deceleration, and idling.
        """
        np.random.seed(101)
        total_steps = int(duration_sec / dt)
        time = np.linspace(0, duration_sec, total_steps)
        
        # Superposition of multiple urban driving frequencies (traffic lights, stops, cruise)
        f1 = 1.0 / 120.0  # Traffic signal frequency
        f2 = 1.0 / 45.0   # Block acceleration
        f3 = 1.0 / 15.0   # Pedal micro-adjustments
        
        base_signal = (
            0.5 * np.sin(2 * np.pi * f1 * time) +
            0.3 * np.sin(2 * np.pi * f2 * time + 0.5) +
            0.2 * np.sin(2 * np.pi * f3 * time + 1.2)
        )
        
        # Shape into realistic EV discharge: mostly positive (discharge), intermittent negative (regen)
        current = np.zeros(total_steps)
        for i in range(total_steps):
            if base_signal[i] > 0.1:
                # Accelerating or cruising (discharge)
                current[i] = (base_signal[i] ** 1.3) * max_current_a
            elif base_signal[i] < -0.3:
                # Strong braking (regen charge, negative current)
                current[i] = base_signal[i] * max_current_a * 0.7
            else:
                # Idling at stops
                current[i] = 0.05 * np.random.uniform(-0.1, 0.1)

        # Smooth slightly to obey motor inertia
        kernel_size = int(3 / dt)
        if kernel_size > 1:
            current = np.convolve(current, np.ones(kernel_size)/kernel_size, mode='same')
            
        return current

    @staticmethod
    def generate_us06(duration_sec=3600, dt=1.0, max_current_a=7.5):
        """
        US06 aggressive highway driving cycle.
        High dynamic peaks, sharp acceleration, and rapid transitions.
        """
        np.random.seed(202)
        total_steps = int(duration_sec / dt)
        time = np.linspace(0, duration_sec, total_steps)
        
        f_high = 1.0 / 25.0
        f_mid = 1.0 / 70.0
        
        signal = 0.6 * np.sin(2 * np.pi * f_mid * time) + 0.4 * np.sin(2 * np.pi * f_high * time)
        spikes = (np.random.uniform(0, 1, total_steps) > 0.92) * np.random.uniform(0.5, 1.0, total_steps)
        
        current = np.zeros(total_steps)
        for i in range(total_steps):
            val = signal[i] + spikes[i]
            if val > 0:
                current[i] = val * max_current_a
            else:
                current[i] = val * max_current_a * 0.8
                
        # Bound limits
        current = np.clip(current, -max_current_a * 0.6, max_current_a)
        return current

    @staticmethod
    def generate_fuds(duration_sec=3600, dt=1.0, max_current_a=4.0):
        """
        Federal Urban Driving Schedule (FUDS) battery loading profile.
        """
        np.random.seed(303)
        total_steps = int(duration_sec / dt)
        t = np.linspace(0, duration_sec, total_steps)
        profile = (
            0.4 * np.sin(2 * np.pi * t / 90.0) +
            0.35 * np.cos(2 * np.pi * t / 40.0) +
            0.25 * np.sin(2 * np.pi * t / 15.0)
        )
        current = np.where(profile >= 0, profile * max_current_a, profile * max_current_a * 0.5)
        return current
