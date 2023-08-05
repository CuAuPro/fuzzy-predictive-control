import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


class ShiftingArray:
    def __init__(self, size):
        self.array = np.zeros(size)
        self.size = size
    
    def add_element(self, new_element):
        self.array[:-1] = self.array[1:]
        self.array[-1] = new_element



def observable_canonical_form(H):
  """Gets the observable canonical form for a system with transfer function H."""
  # Get the poles and coefficients of the transfer function.
  # Digital Control 214
  
  den = np.flip(H.den).reshape(-1,1)
  num = np.flip(H.num).reshape(-1,1)
  nr_poles = len(den) - 1
  
  A_obs = np.eye(nr_poles-1)
  A_right = -den
  A_obs = np.concatenate([np.zeros((1,nr_poles-1)), A_obs],axis=0)
  A_obs = np.concatenate([A_obs, A_right],axis=1)

  #B_obs = num+A_right*num[-1]
  B_obs = num[::-1]
  C_obs = np.zeros((1, nr_poles))
  C_obs[0][-1] = 1
  D_obs = np.array([0])



  model_ss = {}
  model_ss["A"] = A_obs
  model_ss["B"] = B_obs
  model_ss["C"] = C_obs
  model_ss["D"] = D_obs

  return model_ss