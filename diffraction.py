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
        U_screen (array): Array containing the screen complex amplitude function values
        x_screen (array): Screen coordinates in physical units.  Returns angles if U_screen is in angular units.
        F (float): the Fresnel factor (as a check that the Fraunhofer limit is valid)
    """

    U_screen, x_screen, F = _diffraction_1d(U=U, width=width, λ=λ, z=z, fresnel=False)

    if F > 0.5:
        # warn about not in the Fraunhofer regime
        print(f"WARNING: Not in Fraunhofer regime: F={F:.3g}")

    return U_screen, x_screen, F

def fresnel_1d(U, width, λ, z):
    """Calculate the Fraunhofer diffraction pattern for a 1d aperture.

    Args:
        U (array): Array (possibly complex-valued) containing the aperture amplitude function values
        width (float): Width in physical units spanned by U
        λ (float): wavelength in the same units as width
        z (float): distance to the screen in the same units as width (if omitted, the output array will have units of radians)
    
    Returns:
        U_screen (array): Array containing the screen complex amplitude function values
        x_screen (array): Screen coordinates in physical units.
        F (float): the Fresnel factor
    """

    # check for case where z == 0, then no diffraction
    if z==0.0:
        return U, width, np.inf

    return _diffraction_1d(U, width, λ, z=z, fresnel=True)

def _diffraction_1d(U, width, λ, fresnel, z=None):
    """Calculate the diffraction pattern for a 1d aperture.
    This function should not be called directly by users.

    Args:
        U (array): Array (possibly complex-valued) containing the aperture amplitude function values
        width (float): Width in physical units spanned by U
        λ (float): wavelength in the same units as width
        z (float): distance to the screen in the same units as width (if omitted, the output array will have units of radians)
        fresnel (bool): flag to turn on the Fresnel phase correction (otherwise, do a Fraunhofer calculation)
    
    Returns:
        U_screen (array): Array containing the screen complex amplitude function values
        x_screen (array): Screen coordinates in physical units.  Returns angles if U_screen is in angular units.
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
    
    # CHECK the x_s values below

    # array of screen coordinates (centered)
    if z is None: # use angular units for Fraunhofer case if requested
        x_s_temp = np.fft.fftfreq(N_x) * N_x / width * λ # not shifted 
        phase = 1.0 # overall phase (not defined in angular case)
        fade = 1.0 # overall amplitude (not defined in angular case)
    else: # use physical coordinates
        x_s_temp = np.fft.fftfreq(N_x) * N_x * z / width * λ # not shifted 
        phase = -1j*np.exp(1j*k*z)*np.exp(1j*k/2/z*x_s_temp**2) # overall phase
        fade = 1/z/λ # overall amplitude
    amplitude_factor = phase * fade
    
    # FFT kernel function
    if fresnel:
        kernel = np.exp(1j*k/z*(x_a**2))
    else:
        kernel = 1.0

    fft_temp = np.fft.fft(U*kernel)

    # shift the arrays and scale
    U_screen = amplitude_factor * np.fft.fftshift(fft_temp)
    x_screen = np.fft.fftshift(x_s_temp)

    if z is None:
        F = 0 # by assumption
    else:
        est_width = estimate_width_1d(U, width)
        F = est_width**2/z/λ

    return U_screen, x_screen, F

# some helper functions for simple cases

def one_slit(slit_width, array_width, N):
    # TODO find a reasonable default for array_width
    """Aperture function for a single slit.

    Args:
        slit_width (float): width of the slit in physical units
        array_width (float): width of the data array in physical units
        N (int): number of points in the array

    Returns:
        U (array): array containing the slit aperture function values
        x (array): array containing the position values
    """

    x = np.linspace(-array_width/2,array_width/2,N)
    U = (np.abs(x) <= slit_width/2.0)*1.0

    return U, x

def estimate_width_1d(U,width):
    # estimates the effective aperture width
    N = len(U)
    x = np.linspace(0,N,N)
    U2 = np.abs(U)**2
    total = np.sum(U2)
    center_of_mass = np.sum(x*U2)/total
    x_c = x-center_of_mass
    rms = np.sqrt(np.sum(x_c**2*U2)/total)
    correction_factor = np.sqrt(12) # for a uniform slit
    return rms*width/N*correction_factor

def gaussian_beam(beam_radius, array_width, N, λ = None, diagnostics = False):
    """Gaussian beam aperture function.
    Assumes the beam waist is at the aperture.
    
    Args:
        beam_radius (float): the 1/e² beam radius
        array_width (float): the array width in physical units
        N (int): the size of the array
        λ (float): the wavelength (used for diagnostics)
        diagnostics (bool): whether to return diagnostics

    Returns:
        U_a (array): electric field amplitude as fn of position

    Optional returns:
        z_R (float): Rayleigh length
        θ (float): divergence half-angle in radians
        energy (float): total energy of the beam inside the array (should be 1)
    """

    x = np.linspace(-array_width/2, array_width/2, N)
    U_a = np.exp(-x**2/beam_radius**2)

    if not diagnostics:
        return U_a
    
    energy = 2.0/beam_radius**2 *np.sum(np.abs(x)*U_a**2)*array_width/N
    z_R = np.pi*beam_radius**2/λ
    θ = λ/np.pi/beam_radius

    return U_a, z_R, θ, energy