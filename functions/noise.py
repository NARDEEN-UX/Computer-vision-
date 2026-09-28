import numpy as np

def gaussian_noise(img, mean=0, sigma=25):
    # generate gaussian noise matrix
    noise = np.random.normal(mean, sigma, img.shape)

    # add noise to image and clip values
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255)

    return noisy_img.astype(np.uint8)


def snp_noise(img, density=0.05, ratio=0.5):
    noisy_img = img.copy()
    h, w = img.shape[:2]

    # calculate total number of corrupted pixels
    num_noisy = int(density * h * w)

    # split salt and pepper
    num_salt = int(num_noisy * ratio)
    num_pepper = num_noisy - num_salt

    # add salt
    salt_r = np.random.randint(0, h, num_salt)
    salt_c = np.random.randint(0, w, num_salt)
    noisy_img[salt_r, salt_c] = 255

    # add pepper
    pepper_r = np.random.randint(0, h, num_pepper)
    pepper_c = np.random.randint(0, w, num_pepper)
    noisy_img[pepper_r, pepper_c] = 0

    return noisy_img