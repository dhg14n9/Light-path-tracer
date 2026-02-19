import numpy as np
import matplotlib.pyplot as plt

from metrics import Schwarzschild


def pixel_to_viewing_angle(i, n, fov):
    i_unit = (i - n / 2) / (n / 2)
    return np.arctan(i_unit * np.tan(fov / 2))


def get_pixel_color(metric, r_obs, alpha, alpha_crit):
    if alpha < alpha_crit:
        return 0.0
    return 1.0


def main(metric=None):
    if metric is None:
        metric = Schwarzschild(M=1.0)

    width = 800
    height = 800
    fov_deg = 40
    fov = np.radians(fov_deg)

    r_obs = 50.0 * metric.M
    alpha_crit = metric.alpha_crit(r_obs)

    image = np.zeros((width, height))

    for j in range(height):
        for i in range(width):
            alpha_x = pixel_to_viewing_angle(i, width, fov)
            alpha_y = pixel_to_viewing_angle(j, height, fov)
            alpha = np.arccos(np.cos(alpha_x) * np.cos(alpha_y))
            image[i, j] = get_pixel_color(metric, r_obs, alpha, alpha_crit)

    plt.imshow(image, cmap="gray", origin="lower")
    plt.axis("off")
    plt.savefig("black_hole_shadow.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
