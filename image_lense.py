import numpy as np
import matplotlib.image as mpimg

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

def pixel_to_angels(picture_dimension: tuple[int, int], pixel_coordinate: tuple[int, int], fov: float) -> tuple[float, float]:
    """
    Taking the pixel and dimension of the image plane, converting it to angle theta and alpha in radian (an image will be provided later for better understanding)

    Args:
        picture_dimension (tuple(int, int)): dimension of the image plane in (width, height)
        pixel_coordinate (tuple(int, int)): current pixel wanna convert into angle in (i, j)
        fov (float): Field of view angle

    Returns:
        angle (tuple(float, float)): angle viewing the pixel (i, j) in (theta, phi) (in radian)
    """
    width, height = picture_dimension
    i, j = pixel_coordinate

    alpha_x = pixel_to_viewing_angle(i, width, fov)
    alpha_y = pixel_to_viewing_angle(j, height, fov)


    return alpha_x, np.arccos(np.cos(alpha_x) * np.cos(alpha_y))

def angles_to_pixel(picture_dimension: tuple[int, int], angles: tuple[float, float], fov: float) -> tuple[int, int]:
    """
    Taking the angle theta and phi in radian, converting it to pixel coordinate in the image plane (an image will be provided later for better understanding)

    Args:
        picture_dimension (tuple(int, int)): dimension of the image plane in (width, height)
        angles (tuple(float, float)): angle viewing the pixel (i, j) in (theta, alpha) (in radian)
        fov (float): Field of view angle

    Returns:
        pixel_coordinate (tuple(int, int)): current pixel wanna convert into angle in (i, j)
    """
    width, height = picture_dimension
    alpha_x, alpha = angles

    i_unit = np.tan(alpha_x) / np.tan(fov / 2)
    i = int((i_unit + 1) * (width / 2))

    j_unit = np.tan(np.arccos(np.cos(alpha) / np.cos(alpha_x))) / np.tan(fov / 2)
    j = int((j_unit + 1) * (height / 2))

    return j, i


def get_final_ray_angle(r_obs: float, alpha: float) -> float:
    """
    This function takes in the initial angle alpha that the photon from the observer and returns the final angle after the photon escapes. 
    !!! THIS FUNCTION ASSUMES THAT THE PHOTON ESCAPES.

    Args:
        r_obs (float): Distance of the observer from the black hole center
        alpha (float): Initial viewing angle

    Returns:
        float: Final angle after the photon escapes
    """
    
    solution, _ = trace_ray(r_obs, alpha)
    r = solution.y[1] # type: ignore
    phi = solution.y[2] # type: ignore
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    
    return np.arctan2(dy, dx)
    


def get_pixel_color(r_obs: float, alpha: float, theta: float, background_image: np.ndarray, fov: float) -> np.ndarray:
    """
    This function takes in the parameters and determine the color of the background image that the traced ray will hit

    Args:
        r_obs (float): Distance of the observer from the black hole center
        alpha (float): viewing angle (see image for better understanding)
        theta (float): viewing angle (see image for better understanding)
        background_image (np.ndarray): Background image 
        fov (float): Field of view

    Returns:
        np.ndarray: Color of the pixel that the photon hits
    """
    # filter out all the rays that falls into the black hole
    b = viewing_angle_to_impact_parameter(alpha, r_obs)
    if b < B_CRIT:
        return np.array([0.0, 0.0, 0.0])  
    
    final_angle = get_final_ray_angle(r_obs, alpha)
    j_final, i_final = angles_to_pixel(background_image.shape[1::-1], (theta, final_angle), fov)
    
    return background_image[j_final % background_image.shape[0], i_final % background_image.shape[1]]
    


def main(): 
    # load image
    img = mpimg.imread('image.jpg')
    
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = img.astype(np.float32) / 255.0
    
    height, width = img.shape[:2]
    
    # output image 
    lensed_image = np.zeros_like(img)
    
    # obesrver properties
    r_obs = 50.0 * M
    fov_deg = 40.0
    fov: float = np.radians(fov_deg)
    
    for j in range(height):
        for i in range(width):
            lensed_image[j, i] = get_pixel_color(
                r_obs,
                *pixel_to_angels((width, height), (i, j), fov),
                img,
                fov
            )
    
    # save lensed image
    mpimg.imsave('lensed_image.png', lensed_image)



if __name__ == "__main__":
    main()

