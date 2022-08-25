#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# File: beamformer.py
# Author: Irreq
# Date: 21/08-2022

import numpy as np

"""A Time-Domain Semi-Allpass Linear-Phase FIR-based Beamformer."""

def generate_coefficients(n_numtaps=500, save=False):
    """Generate a list containing filter coefficients for all taps."""
    low, high = np.array([1e-10, 0.5-1e-10])  # Generate semi-allpass
    coefficients = []
    for numtaps in range(n_numtaps):
        # Generate oefficients.
        alpha = 0.5 * (numtaps - 1)
        m = np.arange(0, numtaps) - alpha
        h = 0
        h += high * np.sinc(high * m)
        h -= low * np.sinc(low * m)

        if numtaps <= 1:
            window = np.ones(numtaps)
        else:  # Create a hamming window
            # w(n) = 0.54 - 0.46 \cos\left(\frac{2\pi{n}}{numtaps-1}\right)
            #        \qquad 0 \leq n \leq numtaps-1
            fac = np.linspace(-np.pi, np.pi, numtaps)
            window = np.zeros(numtaps)
            for i, k in enumerate([0.54, 0.46]):  # Hamming window
                window += k * np.cos(i * fac)

        h *= window

        scale_frequency = 0.5 * (low + high)
        c = np.cos(np.pi * m * scale_frequency)
        s = np.sum(h * c)
        h /= s
        coefficients.append(h.tolist())
    
    if save:
        with open("coefficients.txt", "w") as f:
            f.writelines(str(coefficients))
            f.close()
    
    return coefficients


def load_coefficients(path):
    """Load the coefficients from a file. Warning large files may create SEGFAULTS."""
    with open(path, "r") as f:
        lines = "".join(f.readlines())
        f.close()
        coefficients = [np.asarray(h) for h in eval(lines)]
    return coefficients


def create_antenna_array(position, element_distance, rows, columns):
    """Generate an array in 3d space with center at (0, 0, 0)."""
    antenna_array = np.zeros((3, rows*columns))

    # Populate matrix
    element_index = 0
    for j in range(rows):
        for i in range(columns):
            antenna_array[0, element_index] = i * element_distance + position[0]
            antenna_array[1, element_index] = j * element_distance + position[1]
            element_index += 1

    # Center matrix in origin (0,0)
    antenna_array[0, :] = antenna_array[0, :] - rows*element_distance/2 + element_distance/2
    antenna_array[1, :] = antenna_array[1, :] - columns*element_distance/2 + element_distance/2
    
    return antenna_array


def get_filter_taps_direction(yaw, pitch, antenna_array, fs, propagation_speed):
    """Calculate taps on antenna for direction.
    
    (yaw, pitch).
    
    Parameters
    ----------
    yaw : float()
        Direction to the sides inside the interval -90.0 < `yaw` < 90.0.
    
    pitch : float()
        Direction up and down inside the interval -90.0 < `pitch` < 90.0.
        Positive values are up and negatives down and 0.0 is straight ahead.

    antenna_array : numpy.ndarray()
        The antenna in 3D space.

    fs : int()
        Sampling-rate for the beamformer.

    propagation_speed : int()
        The speed of the signal, i.e 340m/s for sound traveling in air.

    Returns
    -------

    taps : numpy.ndarray()
        The number of taps for each element on the array in 3D space.
        In the shape of: (`elements`, `ntaps`).
    
    """
    # Invert listen direction
    theta = yaw * -np.pi/180
    phi = pitch * -np.pi/180

    # Calculate delay for each element
    x_factor = np.sin(theta) * np.cos(phi)
    y_factor = np.cos(theta) * np.sin(phi)
    phase_shift_values = antenna_array[0, ] * x_factor + antenna_array[1, ]*y_factor
    phase_shift_values += abs(np.min(phase_shift_values))

    # convert offset in time to ints of taps
    f = lambda offset: offset*2*fs/propagation_speed+1
    taps = f(phase_shift_values).astype(int)

    return taps


def beamform(signals, taps, coefficients):
    """
    Perform a time-domain beamform.
    
    Create a time delay on each element to achieve destructive interference in 
    non-desired directions. This is accomplished with an "allpass" FIR-filter for
    each element. Coefficients are retrieved by a lookup.
    
    Parameters
    ----------
    signals : numpy.ndarray()
        All signals to be beamformed in the shape: (`elements`, `samples`).

    taps : numpy.ndarray()
        Number of taps for each element in the antenna array in the shape: 
        (`elements`, `ntaps`) where `ntaps` are int(). 
    
    coefficients : list()
        A list containing the filter coefficients where the index represents
        number of taps inside that index, i.e. the list is growing in size.

    Returns
    -------
    beamformed : numpy.array()
        The beamformed 1D data in the shape of: (`samples`,).

    References
    ----------
    scipy.signal.firwin, scipy.signal.lfilter
    """
    
    beamformed = np.empty_like(signals)

    for element_index, (numtaps, stream) in enumerate(list(zip(taps, signals))):
        h = coefficients[numtaps]
        out_full = np.apply_along_axis(lambda y: np.convolve(h, y), 0, stream)

        # Reshape data
        beamformed[element_index] = out_full[:stream.shape[0]]

    return np.sum(beamformed, axis=0) / beamformed.shape[0]

        

