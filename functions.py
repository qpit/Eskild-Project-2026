import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib as jl
import numpy as np

def refractive_index_sellmeier(A, B, C, D, E, F, wavelength):
    """
    Returns the refractive index using the Sellmeier equation.
    Parameters:
    A, B, C, D, E, F : float
        Sellmeier coefficients (Ensure these match the micrometer scale).
    wavelength : float or np.ndarray
        Wavelength in nanometers.   
    Returns:
        float or np.ndarray
            Refractive index. Returns np.nan for wavelengths outside 
            the valid transparency range (where n^2 < 0).
    """
    lam_um = np.array(wavelength) * 1e-3 # Convert nm to micrometers
    lam_sq = lam_um**2
    # n_z^2 = A + B/(1 - C/L^2) + D/(1 - E/L^2) - F*L^2
    n_squared = (A + 
                 (B / (1 - (C / lam_sq))) + 
                 (D / (1 - (E / lam_sq))) - 
                 F * lam_sq)
    # Create a mask for valid (non-negative) n^2 values
    valid_mask = n_squared >= 0
    n_val = np.full_like(n_squared, np.nan)
    # Only take the square root of valid (positive) numbers
    np.putmask(n_val, valid_mask, np.sqrt(n_squared, where=valid_mask, out = np.full_like(n_squared, np.nan)))
    # If the input was a single float, return a float, otherwise return array
    if n_val.ndim == 0:
        return float(n_val)
    return n_val
def refractive_index_poly(T, a1_list, a2_list, wavelength):
    """
    Returns the refractive index using a polynomial temperature dependence at reference temperature of 25°C.
    Parameters:
    T : float or np.ndarray
        Temperature in Kelvin.
    a1_list, a2_list : list of float
        Coefficients for the polynomial temperature dependence. They should be lists of length 4.
    wavelength : float or np.ndarray
        Wavelength in nanometers.
    """
    #Convert to micrometers
    wavelength = wavelength * 1e-3 #Convert to micrometers
    T = T-273.15  # Convert to Celsius
    term1 = (T-25) * (a1_list[0] + a1_list[1]/wavelength + a1_list[2]/(wavelength**2) + a1_list[3]/(wavelength**3))
    term2 = (T-25)**2 * (a2_list[0] + a2_list[1]/wavelength + a2_list[2]/(wavelength**2) + a2_list[3]/(wavelength**3))
    return term1 + term2
def refractive_index_total(T, A,B,C,D,E,F,a1_list,a2_list,wavelength):
    """
    Returns the total refractive index considering both Sellmeier equation and polynomial temperature dependence.
    Parameters:
    T : float or np.ndarray
        Temperature in Kelvin.
    A, B, C, D, E, F : float
        Sellmeier coefficients for the material.
    a1_list, a2_list : list of float
        Coefficients for the polynomial temperature dependence. They should be lists of length 4.
    wavelength : float or np.ndarray
        Wavelength in nanometers.
    Returns:
        float or np.ndarray
            Total refractive index at the given temperature(s) and wavelength(s).
    """
    term1 = refractive_index_sellmeier(A,B,C,D,E,F,wavelength)
    term2 = refractive_index_poly(T,a1_list,a2_list,wavelength)
    return term1 + term2
def wave_vector(n, omega):
    """
    Calculate the wave vector k given refractive index n and angular frequency omega.
    Parameters:
    n : float or np.ndarray
        Refractive index of the medium.
    omega : float or np.ndarray
        Angular frequency (rad/s).
    Returns:
        k : float or np.ndarray
            Wave vector (1/m).
    """
    c = 299792458  # Speed of light in m/s
    k = n * omega / c
    return k
def phase_mismatch(n_2w, omega_2w, n_w, omega_w):
    """
    Calculate the phase mismatch Δk for second harmonic generation (SHG).
    Parameters:
    n_2w : float or np.ndarray
        Refractive index at the second harmonic frequency.
    omega_2w : float or np.ndarray
        Angular frequency at the second harmonic (rad/s).
    n_w : float or np.ndarray
        Refractive index at the fundamental frequency.
    omega_w : float or np.ndarray
        Angular frequency at the fundamental (rad/s).
    Returns:
        Delta_k : float or np.ndarray
            Phase mismatch (1/m).
    """
    return wave_vector(n_2w, omega_2w) - 2 * wave_vector(n_w, omega_w)
def poling_period(Lambda_0, T, T_0, a_z):
    """
    Calculate the poling period Λ(T) as a function of temperature.
    Parameters:
    Lambda_0 : float
        Poling period at reference temperature T_0 (in meters).
    T : float
        Current temperature (in Kelvin).
    T_0 : float
        Reference temperature (in Kelvin).
    a_z : float
        Thermal expansion coefficient along the z-axis (1/K).
    Returns:
        Lambda_T : float
    """
    return Lambda_0 * (1 + a_z * (T - T_0))
def quasi_phase_mismatch(n_2w, omega_2w, n_w, omega_w, Lambda_0, T, T_0, a_z):
    """
    Calculate the quasi-phase mismatch Δk_QPM.
    Parameters:
    Delta_k : float or np.ndarray
        Phase mismatch (1/m).
    Lambda_T : float
        Poling period at temperature T (in meters).
    Returns:
        Delta_k_QPM : float or np.ndarray
            Quasi-phase mismatch (1/m).
    """
    Delta_k = phase_mismatch(n_2w, omega_2w, n_w, omega_w)
    Lambda_T = poling_period(Lambda_0, T, T_0, a_z)
    return Delta_k - (2 * np.pi) / Lambda_T
def second_harmonic_output(QPM, L):
    """
    Calculate the second harmonic output amplitude as a function of QPM wavevector mismatch and crystal length.
    Parameters:
    QPM : float
        The wavevector mismatch (in 1/m).
    L : float
        The length of the nonlinear crystal (in m).
    Returns:
    float
        The second harmonic output amplitude.
    """
    term = L*np.exp(1j*QPM*L/2)*np.sinc(QPM*L/2/np.pi)
    real = np.real(term)
    imag = np.imag(term)
    amplitude = np.sqrt(real**2 + imag**2)
    return amplitude
def complex_second_harmonic_field(QPM, L):
    """
    Calculate the complex second harmonic output field.
    Parameters:
    QPM : float
        The wavevector mismatch (in 1/m).
    L : float
        The length of the nonlinear crystal (in m).
    Returns:
    complex
        The second harmonic output complex field.
    """
    return L*np.exp(1j*QPM*L/2)*np.sinc(QPM*L/2/np.pi)

def double_pass_SHG(QPM, L, phi_reflec):
    """
    Calculate the second harmonic output for a double pass through the crystal.
    Parameters:
    QPM : array_like
        Quasi-phase mismatch values.
    L : float
        Length of the crystal.
    phi_reflec : float
        Phase shift upon reflection.
    """
    term1 = second_harmonic_output(QPM,L)
    term2 = 1+np.exp(1j*(phi_reflec+L*QPM))
    term2_real = np.real(term2)
    term2_imag = np.imag(term2)
    term2_amp = np.sqrt(term2_real**2 + term2_imag**2)
    return term1*term2_amp
def SH_field_left(QPM, L):
    """
    Calculate the field strength after reflection at end of crystal. Meaning integrating from L to 0.
    Parameters:
    QPM : array
        Quasi-phase mismatch values.
    L : float
        Length of the crystal.
    Returns:
    term : array
        Second harmonic field strength after reflection.
    """
    term = -complex_second_harmonic_field(QPM, L)
    return term
def SH_field_right(QPM, L,Delta_r):
    """
    ????? Physical?
    Parameters:
    QPM : array
        Quasi-phase mismatch values.
    L : float
        Length of the crystal.
    Delta_r : float
        Edge width at the right end of the crystal.
    Returns:
    term : array
        ?????? Physical?
    """
    return L * np.exp(1j * QPM * ( 3/2*L+2*Delta_r)) * np.sinc(QPM * L / 2 / np.pi)
def final_cavity_field(tau, rho, QPM, L, Delta_r, Delta_l, D, n_2w, omega_2w, n_w, omega_w):
    """
    Calculate the final second harmonic field exiting the cavity.
    Parameters:
    tau : float
        Transmission coefficient of the output coupler.
    rho : float
        Reflection coefficient of the output coupler.
    QPM : float
        Quasi-phase mismatch (1/m).
    L : float
        Length of the crystal (m).
    Delta_r : float
        Edge width at the right end of the crystal (m).
    Delta_l : float
        Edge width at the left end of the crystal (m).
    D : float
        Distance between the crystal end and the output coupler (m).
    n_2w : float
        Refractive index at the second harmonic frequency.
    omega_2w : float
        Angular frequency at the second harmonic (rad/s).
    n_w : float
        Refractive index at the fundamental frequency.
    omega_w : float
        Angular frequency at the fundamental (rad/s).
    Returns:
    float
        Final second harmonic field amplitude exiting the cavity.
    """
    field_left = SH_field_left(QPM, L)
    field_right = SH_field_right(QPM, L, Delta_r)
    phase_mismatch_value = phase_mismatch(n_2w, omega_2w, n_w, omega_w)
    wave_vector_2w = wave_vector(n_2w, omega_2w)
    numerator = 1j*tau*(field_left*np.exp(1j*(wave_vector_2w*D-Delta_l*phase_mismatch_value)) + field_right*np.exp(-1j*Delta_r*phase_mismatch_value))
    denominator = np.exp(2*1j*wave_vector_2w*D)-rho
    real_val = np.real(numerator/denominator)
    imag_val = np.imag(numerator/denominator)
    return np.sqrt(real_val**2 + imag_val**2)

