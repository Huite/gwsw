import numpy as np


def c_entry(L, B, c0):
    """Ernst & Van Bakel: stream-bed entry resistance term."""
    return L / B * c0

def c_vertical(D, kv):
    """
    De Lange: vertical resistance term.
    """
    return 0.5 * D / kv

def c_horizontal(L, D, kh):
    """Ernst: horizontal resistance term."""
    return (2.0 / 3.0) * L**2 / (8.0 * kh * D)

def c_radial(L, B, D, kh, kv):
    """Ernst: radial resistance term, includes anisotropy."""
    c = (
        L
        / (np.pi * np.sqrt(kh * kv))
        * np.log((4.0 * D) / (np.pi * B) * np.sqrt(kh / kv))
    )
    c[c < 0] = 0.0
    return c

def c_ernst(L, B, D, kh, kv, c0):
    """
    Ernst: serial drainage resistance for a single layer.
    For use in discretized models: use ``L=np.minimum(cell_width, L + B)``.

    Parameters
    ----------
    L :
        distance between drains, or cell width (m)
        Excludes width of water body!
    B :
        wetted perimeter (m)
    D :
        saturated thickness (m)
    kh :
        horizontal conductivity (m/d)
    kv :
        vertical conductivity (m/d)
    c0 :
        entry resistance (d)

    Returns
    -------
    c :
        cell drain resistance (d)
    """
    return (
        c_vertical(D, kv)
        + c_horizontal(L, D, kh)
        + c_radial(L, B, D, kh, kv)
        + c_entry(L, B, c0)
    )


def coth(x):
    """Functin: hyperbolic cotangent."""
    e2x = np.exp(2.0 * x)
    return (e2x + 1.0) / (e2x - 1.0)


def c_horizontal_de_lange(L, B, D, kh, kv, c0, c1):
    def f(x):
        return x * coth(x)

    fraction_wetted = B / (B + L)
    fraction_land = L / (B + L)
    # Compute total resistance to aquifer; requires full thickness (D).
    c1_prime = c1 + (D / kv)
    # Compute labda for below-ditch part (B) and land (L) part
    labda_B = np.sqrt((kh * D * c1_prime * c0) / (c1_prime + c0))
    labda_L = np.sqrt(c1_prime * kh * D)
    # x=0 is located at water-land interface, so x_B is negative and x_L is
    # positive.
    x_B = -B / (2.0 * labda_B)
    x_L = L / (2.0 * labda_L)
    c_L = (c0 + c1_prime) * f(x_L) + (c0 * L / B) * f(x_B)
    c_B = (c1_prime + c0) / (1.0 - L / B * c0 / c_L)
    # Weight resistance by faction wetted/land.
    c_total = 1.0 / (fraction_wetted / c_B + fraction_land / c_L)
    return c_total - c1_prime

def c_vertical_de_lange(L, B, D, kh, kv, c0, c1):
    # Compute total resistance to aquifer; requires full thickness (D).
    c1_prime = c1 + (D / kv)
    # Compute labda for below-ditch part (B) and land (L) part
    labda_B = np.sqrt((kh * D * c1_prime * c0) / (c1_prime + c0))
    labda_L = np.sqrt(c1_prime * kh * D)
    # Compute maximum path lengths
    horizontal_length_L = np.minimum(3 * labda_L, L)
    horizontal_length_B = np.minimum(3 * labda_B, B)
    c_vertical = D / kv
    c_vertical_L = np.minimum(horizontal_length_L / kh, c_vertical)
    c_vertical_B = np.minimum(horizontal_length_B / kh, c_vertical)
    return (c_vertical_L + c_vertical_B) / 2.0

def c_de_lange_2022(L, B, D, kh, kv, c0, c1):
    """
    De Lange 2020 two-layer cell drainage resistance with anisotropic radial resistance.
    For use in discretized models: use ``L=np.minimum(cell_width, L + B)``.

    Parameters
    ----------
    L :
        distance between drains, or cell width (m), excluding water!
    B :
        wetted perimeter (m)
    D :
        saturated thickness (m)
    kh :
        horizontal conductivity (m/d)
    kv :
        vertical conductivity (m/d)
    c0 :
        entry resistance (d)
    c1 :
        aquitard resistance (d)

    Returns
    -------
    c :
        cell drain resistance (d)
    """

    def effective_perimeter(B, kh, D, c0, c1):
        c1_prime = c1 + (D / kv)
        labda_B = np.sqrt((kh * D * c1_prime * c0) / (c1_prime + c0))
        a = B / (2.0 * labda_B)
        F_B = a + 1.0 / (1.0 + a)
        return B / F_B

    B_eff = effective_perimeter(B, kh, D, c0, c1)
    return (
        c_vertical_de_lange(L, B, D, kh, kv, c0, c1)
        + c_horizontal_de_lange(L, B, D, kh, kv, c0, c1)
        + c_radial(L, B_eff, D, kh, kv)
    )
