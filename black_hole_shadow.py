import numpy as np
import matplotlib.pyplot as plt

from geodesic_tracer import (
    M, R_S, R_PHOTON, B_CRIT,
    trace_ray, viewing_angle_to_impact_parameter
)



def pixel_to_viewing_angle(i: int, n: int, fov: float) -> float:
    """
    Taking the pixel and dimension of the image plane, converting it to viewing angle in a specific direction (x, y)

    Args:
        i (int): current pixel wanna convert into angle
        n (int): dimension of the image plane in the direction of i
        fov (float): Field of view angle

    Returns:
        angle (float): angle viewing the pixel i (in radian)
    """
    
    i_unit: float = (i - n/2) / (n/2)
    return np.arctan(i_unit * np.tan(fov / 2))


def get_pixel_color(r_obs: float, alpha: float) -> float:
    """
    Get the pixel color correspond to a color (Just for modulization, kinda verbose if generating a mere black and white shadow image)

    Args:
        r_obs (float): observer's distance from black hole's center
        alpha (float): viewing angle

    Returns:
        pixel color (float): a value correspond to pixel color. For now, 
            - 0: light ray goes inside the black hole
            - 1: light ray escape
    """
    
    # debugging info
    # print(f"Tracing ray at alpha = {alpha:.4f} rad...")
    
    # filter out all the rays that falls into the black hole
    b = viewing_angle_to_impact_parameter(alpha, r_obs)
    if b < B_CRIT:
        return 0.0
    
    # _, outcome = trace_ray(r_obs, alpha)
    return 1.0 # if outcome == "escaped" else 0.0


def main(): 
    # image plane property
    width = 800
    height = 800
    fov_deg = 40
    fov = np.radians(fov_deg)

    # observer position
    r_obs = 50.0 * M

    # creating an image 
    image = np.zeros((width, height))
    
    # loop through the image plane
    for j in range(height): 
        for i in range(width):
            # debugging info
            # print(f"Processing pixel ({i}, {j})...")
            
            alpha_x = pixel_to_viewing_angle(i, width, fov)
            alpha_y = pixel_to_viewing_angle(j, height, fov)
            alpha = np.arccos(np.cos(alpha_x) * np.cos(alpha_y))


            image[i, j] = get_pixel_color(r_obs, alpha)

    # export image
    plt.imshow(image, cmap="gray", origin="lower")
    plt.axis("off")
    plt.savefig("black_hole_shadow.png", dpi=200, bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    main()

