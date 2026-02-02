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