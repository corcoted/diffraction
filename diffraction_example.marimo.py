import marimo

__generated_with = "0.7.13"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    # single slit diffraction
    slit_width_slider = mo.ui.slider(1,1000, label="slit width (μm)")
    slit_width_slider
    return slit_width_slider,


@app.cell
def __(U_a, mo, plt, x_a):
    # plot!
    _fig, _ax = plt.subplots(figsize=(5,3))
    _ax.plot(x_a,U_a)
    _ax.set_xlabel("position (mm)")
    _ax.set_title("aperture function")
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def __(I_s, U_s, dif, mo, np, plt, x_s):
    # plot!
    # the factors of 1000 are to switch x units to milliradians for plotting

    # guess some limits for the x axis
    _a_est = dif.estimate_width_1d(U_s, np.max(x_s) - np.min(x_s)) * 1000

    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.plot(x_s * 1000, I_s)
    _ax.set_xlabel("field angle (mrad)")
    _ax.set_title("diffraction intensity")
    _ax.set_xlim(-_a_est, _a_est)
    mo.mpl.interactive(_fig)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Calculations go below""")
    return


@app.cell
def __():
    import marimo as mo
    import diffraction as dif
    import numpy as np
    import matplotlib.pyplot as plt
    return dif, mo, np, plt


@app.cell
def __():
    # constants (using millimeters for length units)
    λ = 532e-6 # wavelength
    N_x = 2**12 # size of arrays
    slit_width_multiplier = 2**6
    return N_x, slit_width_multiplier, λ


@app.cell
def __(slit_width_multiplier, slit_width_slider):
    slit_width = slit_width_slider.value / 1000.
    array_width = slit_width*slit_width_multiplier
    return array_width, slit_width


@app.cell
def __(N_x, array_width, dif, slit_width):
    # calculate the aperture function
    U_a, x_a = dif.one_slit(slit_width, array_width, N_x)
    return U_a, x_a


@app.cell
def __(U_a, array_width, dif, np, λ):
    # far field diffraction in angular units
    U_s, x_s, F = dif.fraunhofer_1d(U_a, array_width, λ)
    I_s = np.abs(U_s)**2
    return F, I_s, U_s, x_s


@app.cell
def __(mo):
    mo.md(r"""### Scratch goes below""")
    return


@app.cell
def __(mo):
    mo.md(r"""#### Testing linked sliders""")
    return


@app.cell
def __(mo):
    get_c, set_c = mo.state(0)
    return get_c, set_c


@app.cell
def __(get_c, mo, set_c):
    c = mo.ui.slider(0,100,0.1, value=get_c(), on_change=set_c, label="Temp (C)")
    return c,


@app.cell
def __(get_c, mo, set_c):
    f = mo.ui.slider(32,212,0.18, value=get_c()*9./5.+32., on_change=lambda x: set_c((x-32.)*5./9.), label="Temp (F)")
    return f,


@app.cell
def __(c, f):
    [c, f]
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
