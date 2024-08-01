# functions for calculating 1D and 2D diffraction patterns
# given an array containing the aperture function,
# using the Fourier transform method

import numpy as np

def fraunhofer_1d(U, width, λ, z=None):
    """Calculate the Fraunhofer diffraction pattern for a 1d aperture.

    Args:
        U (array): Array (possibly complex-valued) containing the aperture amplitude function values
        width (float): Width in physical units spanned by U
        λ (float): wavelength in the same units as width
        z (float): distance to the screen in the same units as width (if omitted, the output array will have units of radians)
    
    Returns:
        U_screen (array): Array containing the screen amplitude function values
        w_screen (float): Width of U_screen in physical units.  Returns "None" if U_screen is in angular units.
        F (float): the Fresnel factor (as a check that the Fraunhofer limit is valid)
    """
    return _diffraction_1d(U, width, λ, z, fresnel=False)

def fresnel_1d(U, width, λ, z):
    """Calculate the Fraunhofer diffraction pattern for a 1d aperture.

    Args:
        U (array): Array (possibly complex-valued) containing the aperture amplitude function values
        width (float): Width in physical units spanned by U
        λ (float): wavelength in the same units as width
        z (float): distance to the screen in the same units as width (if omitted, the output array will have units of radians)
    
    Returns:
        U_screen (array): Array containing the screen amplitude function values
        w_screen (float): Width of U_screen in physical units.  Returns "None" if U_screen is in angular units.
        F (float): the Fresnel factor
    """

    return _diffraction_1d(U, width, λ, z, fresnel=True)

def _diffraction_1d(U, width, λ, z=None, fresnel):
    """Calculate the diffraction pattern for a 1d aperture.
    This function should not be called directly by users.

    Args:
        U (array): Array (possibly complex-valued) containing the aperture amplitude function values
        width (float): Width in physical units spanned by U
        λ (float): wavelength in the same units as width
        z (float): distance to the screen in the same units as width (if omitted, the output array will have units of radians)
        fresnel (bool): flag to turn on the Fresnel phase correction (otherwise, do a Fraunhofer calculation)
    
    Returns:
        U_screen (array): Array containing the screen amplitude function values
        w_screen (float): Width of U_screen in physical units.  Returns "None" if U_screen is in angular units.
        F (float): the Fresnel factor
    """

    # Check for valid inputs
    if width<=0.0: # must be positive
        raise(ValueError)
    if λ<=0.0: # must be positive
        raise(ValueError)
    
    N_x = len(U)
    if N_x < 2: # then U is not an array
        raise(ValueError)

    if ((z is None) and fresnel): # fresnel requires a screen distance
        raise(ValueError)

    # definitions
    k = 2.0*np.pi/λ

    # array of aperture coodinates (centered)
    x_a = np.arange(N_x) * width/N_x - width/2.0
    
    # array of screen coordinates (centered)
    if z is None: # use angular units for Fraunhofer case if requested
        x_s = np.arange(N_x) / width / k
    else: # use physical coordinates
        x_s = np.arange(N_x) * z / width / k
    
    return
