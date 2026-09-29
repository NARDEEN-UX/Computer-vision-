from backend.app.algorithms.noise.gaussian import gaussian_noise
from backend.app.algorithms.noise.salt_pepper import salt_pepper_noise


def snp_noise(img, density=0.05, ratio=0.5, seed=None):
    """Backward-compatible name: ratio maps to salt_probability."""
    return salt_pepper_noise(img, density=density, salt_probability=ratio, seed=seed)
