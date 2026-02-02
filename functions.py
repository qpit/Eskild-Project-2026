def refractive_index_sellmeier(A,B,C,D,E,F,wavelength):
    """
    Returns the refractive index using the Sellmeier equation with a reference temperature of 25°C.
    Parameters:
    A, B, C, D, E, F : float
        Sellmeier coefficients for the material.
    wavelength : float or np.ndarray
        Wavelength in nanometers.
    Returns:
        float or np.ndarray
            Refractive index at the given wavelength(s).
    """
    #To SI units (meters)
    wavelength = wavelength * 1e-9
    term = A + (B/(1-(C/(wavelength**2)))) + (D/(1-(E/(wavelength**2)))) - F*wavelength**2
    return np.sqrt(term)
def refractive_index_poly(T, a1_list, a2_list, wavelength):
    """
    Returns the refractive index using a polynomial temperature dependence at reference temperature of 25°C.
    Parameters:
    T : float or np.ndarray
        Temperature in degrees Celsius.
    a1_list, a2_list : list of float
        Coefficients for the polynomial temperature dependence. They should be lists of length 4.
    wavelength : float or np.ndarray
        Wavelength in nanometers.
    """
    #To SI units (meters)
    wavelength = wavelength * 1e-9 #Convert to meters
    T = T-273.15  # Convert to Kelvin
    term1 = (T-25) * (a1_list[0] + a1_list[1]/wavelength + a1_list[2]/(wavelength**2) + a1_list[3]/(wavelength**3))
    term2 = (T-25)**2 * (a2_list[0] + a2_list[1]/wavelength + a2_list[2]/(wavelength**2) + a2_list[3]/(wavelength**3))
    return term1 + term2
def refractive_index_total(T, A,B,C,D,E,F,a1_list,a2_list,wavelength):
    """
    Returns the total refractive index considering both Sellmeier equation and polynomial temperature dependence.
    Parameters:
    T : float or np.ndarray
        Temperature in degrees Celsius.
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