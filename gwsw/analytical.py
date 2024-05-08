"""
Analytical approaches
=====================

Note: `a` is used a temporary variable here and there.
"""

import numpy as np

# MODFLOW
# -------
#
# The approach as described in the MODFLOW manual for the RIVER package only
# considers a hydraulic resistance of the riverbed material.


def c_entry(L, B, c0):
    """
    Parameters
    ----------
    L :
        distance between drains, or cell width (m)
        Excludes width of water body!
    B :
        wetted perimeter (m)
    c0 :
        entry resistance (d)

    Returns
    -------
    c_entry :
        drain entry resistance
    """
    return L / B * c0


def c_modflow(L, B, c0):
    """
    Compute a cell drain resistance, as prescribed by the RIVER section in the
    MODFLOW manual.

    Parameters
    ----------
    L :
        distance between drains, or cell width (m)
        Excludes width of water body!
    B :
        wetted perimeter (m)
    c0 :
        entry resistance (d)

    Returns
    -------
    c :
        cell drain resistance (d)
    """
    return c_entry(L, B, c0)


# Ernst
# -----
#
# Ernst's approach consist of breaking down the resistance within a cell to a
# horizontal, a vertical, and a radial resistance. A drain entry resistance is
# commonly added as well.


def c_vertical(D, kv):
    """
    Ernst's vertical resistance.

    Note: Ernst uses a D*, which is "modified" thickness.

    Parameters
    ----------
    D:
        saturated thickness (m)
    kv:
        vertical conductivity (m/d)

    Returns
    -------
    c_vertical
        vertical resistance (d)
    """
    return D / kv


def c_horizontal(L, D, kh):
    """
    Ernst's horizontal resistance.

    Contains a "form factor" of 2/3 since we are using the average head.

    Parameters
    ----------
    L :
        distance between drains, or cell width (m)
        Excludes width of water body!
    kh :
        horizontal conductivity (m/d)
    kv :
        vertical conductivity (m/d)
    B :
        wetted perimeter (m)
    D :
        saturated thickness (m)

    Returns
    -------
    c_horizontal
        horizontal resistance (d)
    """
    return (2.0 / 3.0) * L**2 / (8.0 * kh * D)


def c_radial(L, B, D, kh, kv):
    """
    Ernst's (simplified) radial resistance term to a drain, with anisotropy
    correction.

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

    Returns
    -------
    c_radial :
        radial resistance (d)
    """
    c = (
        L
        / (np.pi * np.sqrt(kh * kv))
        * np.log((4.0 * D) / (np.pi * B) * np.sqrt(kh / kv))
    )
    #    c = c.where(~(c < 0.0), other=0.0)
    c[c < 0] = 0.0
    return c


def c_ernst(L, B, D, kh, kv, c0):
    """
    Ernst's drainage resistance with anisotropic radial resistance, and including
    entry resistance.

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


# "Ernst++"
# ---------
#
# Flow under a wide ditch varies in strength: the head difference is strongest
# at the side of the ditch weakens towards the middle (cf. Mazure). This may be
# represented by a reduced effective wet perimeter, with a constant flow
# strength. This reduced perimeter is used for the entry and radial resistance.


def effective_perimeter(B, labda_B):
    """
    Compute an effective wetted perimeter through which the groundwater flows.

    Parameters
    ----------
    B :
        wetted perimeter (m)
    labda_B :
        leakage factor for the ditch part (m)

    Returns
    -------
    B_eff :
        Effective wetted perimeter (m)
    """
    a = B / (2.0 * labda_B)
    F_B = a + 1.0 / (1.0 + a)
    return B / F_B


def coth(x):
    """Hyperbolic cotangent"""
    e2x = np.exp(2.0 * x)
    return (e2x + 1.0) / (e2x - 1.0)


def c_horizontal_multilayer(L, D, kh, c1):
    """
    Horizontal resistance, including the influence of a transmissive aquifer
    below, separated by an aquitard.

    Parameters
    ----------
    L :
        distance between drains, or cell width (m)
        Excludes width of water body!
    D :
        saturated thickness (m)
    kh :
        horizontal conductivity (m/d)
    c1 :
        aquitard resistance (d)

    Returns
    -------
    c_horizontal
        horizontal resistance (d)
    """
    a = L / (2.0 * np.sqrt(kh * D * c1))
    return c1 * a * coth(a) - c1


def c_ernst_multilayer(L, B, D, kh, kv, c0, c1):
    """
    After De Lange, includes the effect of a deeper transmissive aquifer, and
    the variation of flow under the ditch.

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
    c1 :
        aquitard resistance (d)

    Returns
    -------
    c :
        cell drain resistance (d)
    """
    return (
        c_vertical(D, kv)
        + c_horizontal_multilayer(L, D, kh, c1)
        + c_radial(L, B, D, kh, kv)
        + c_entry(L, B, c0)
    )


def c_ernst_multilayer_no_vertical(L, B, D, kh, kv, c0, c1):
    """
    After De Lange, includes the effect of a deeper transmissive aquifer, and
    the variation of flow under the ditch.

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
    c1 :
        aquitard resistance (d)

    Returns
    -------
    c :
        cell drain resistance (d)
    """
    return (
        c_horizontal_multilayer(L, D, kh, c1)
        + c_radial(L, B, D, kh, kv)
        + c_entry(L, B, c0)
    )


# De Lange
# --------


def c_de_lange_1997(L, B, D, kh, kv, c0, c1):
    """
    De Lange cell drainage resistance with anisotropic radial resistance.

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

    def f(x):
        return x * coth(x)

    fraction_wetted = B / (B + L)
    # Compute total resistance to aquifer
    c1_prime = c1 + (D / kv)
    # Compute labda for below-ditch part (B) and land (L) part
    labda_B = np.sqrt((kh * D * c1_prime * c0) / (c1_prime + c0))
    labda_L = np.sqrt(c1_prime * kh * D)

    # x=0 is located at water-land interface, so x_B is negative and x_L is
    # positive.
    x_B = -B / (2.0 * labda_B)
    x_L = L / (2.0 * labda_L)

    c_rad = c_radial(L, B, D, kh, kv)
    c_L = (c0 + c1_prime) * f(x_L) + (c0 * L / B) * f(x_B)
    c_B = (c1_prime + c0) / (c_L - c0 * L / B) * c_L
    c_total = 1.0 / (fraction_wetted / c_B + (1.0 - fraction_wetted) / c_L) + c_rad

    # Subtract aquifer and aquitard resistance from feeding resistance
    c = c_total - c1_prime
    return c


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
    c_B = (c1_prime + c0) / (c_L - c0 * L / B) * c_L
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
    De Lange cell drainage resistance with anisotropic radial resistance.

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

    # TODO: Cell size smaller than L?
    # Add cell_size argument.
    # TODO: Effective D for horizontal, radial

    B_eff = effective_perimeter(B, kh, D, c0, c1)
    return (
        c_vertical_de_lange(L, B, D, kh, kv, c0, c1)
        + c_horizontal_de_lange(L, B, D, kh, kv, c0, c1)
        + c_radial(L, B_eff, D, kh, kv)
    )
