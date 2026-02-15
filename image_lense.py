# Every set of coordinate is written as (y, x)
# FOV pairs: (horizontal, vertical)

import numpy as np
import matplotlib.image as mpimg
from tqdm import tqdm

from geodesic_tracer import (
    M, R_S, R_PHOTON, B_CRIT,
    trace_ray, viewing_angle_to_impact_parameter
)

def pixel_to_angles(pixel: tuple[int, int], image_dimension: tuple[int, int], fov: tuple[float, float]) -> tuple[float, float]:
    """
    This function takes in the pixel coordinates, the image dimensions, and fov and
    returns the corresponding angles (alpha, theta) for the given pixel.
    
    Args:
        pixel (tuple[int, int]): The coordinate of the pixel in the image
        image_dimension (tuple[int, int]): Dimension of the image (height, width)
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical)

    Returns:
        tuple[float, float]: The corresponding angles (alpha, theta) for the given pixel
    """
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    x = pixel[1] - width / 2
    y = pixel[0] - height / 2

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    x_cam = x / fx
    y_cam = y / fy

    alpha = np.arctan2(np.sqrt(x_cam**2 + y_cam**2), 1.0)
    theta = np.arctan2(x_cam, y_cam)
    return (alpha, theta)


def angles_to_pixel(angles: tuple[float, float], image_dimension: tuple[int, int], fov: tuple[float, float], clip: bool = False) -> tuple[int, int]:
    """
    Inverse of ``pixel_to_angles``: convert (alpha, theta) back to pixel (y, x).

    Args:
        angles (tuple[float, float]): (alpha, theta) in radians.
        image_dimension (tuple[int, int]): Image dimension (height, width).
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical).
        clip (bool): Clamp output pixel to image bounds if True.

    Returns:
        tuple[int, int]: Pixel coordinate (y, x).
    """
    alpha, theta = angles
    height, width = image_dimension
    horizontal_fov, vertical_fov = fov

    fx = (width / 2) / np.tan(horizontal_fov / 2)
    fy = (height / 2) / np.tan(vertical_fov / 2)

    r_cam = np.tan(alpha)
    x_cam = r_cam * np.sin(theta)
    y_cam = r_cam * np.cos(theta)

    x = x_cam * fx
    y = y_cam * fy

    px = int(np.rint(x + width / 2))
    py = int(np.rint(y + height / 2))

    if clip:
        # TODO: This clipping is not yet perfect (I think)
        px = int(np.clip(px, 0, width - 1))
        py = int(np.clip(py, 0, height - 1))
        pass

    return (py, px)



ray_cache = {}

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
    
    # check cache 
    key = round(alpha, 5)
    if key in ray_cache:
        return ray_cache[key]
    
    solution, _ = trace_ray(r_obs, alpha)
    r = solution.y[1] # type: ignore
    phi = solution.y[2] # type: ignore
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    
    dx = x[-1] - x[-2]
    dy = y[-1] - y[-2]
    
    ray_cache[key] = np.arctan2(dy, dx)
    
    return np.arctan2(dy, dx)


#! As I implement the alpha first look up table this function will become obsolete
def get_color_from_background(angles: tuple[float, float], background_image: np.ndarray, fov: tuple[float, float], r_obs: float) -> np.ndarray:
    """
    This function takes in the final angles (alpha, theta) of the photon after escaping and returns the corresponding color from the background image.

    Args:
        angles (tuple[float, float]): (alpha, theta) in radians (initial angles from the observer).
        background_image (np.ndarray): Background image array.
        fov (tuple[float, float]): Field of view in radians (horizontal, vertical).

    Returns:
        np.ndarray: Color of the pixel in the background image corresponding to the given angles.
    """
    
    # check if the rays will actually escape the black hole
    alpha, theta = angles
    b = viewing_angle_to_impact_parameter(alpha, r_obs)
    if b < B_CRIT:
        return np.array([0.0, 0.0, 0.0])
    
    final_angle = get_final_ray_angle(r_obs, alpha)
    final_pixel = angles_to_pixel((final_angle, theta), background_image.shape[:2], fov)
    
    # for debugging purposes, returns a specific color if the ray escapes but goes out of bounds of the background image
    if final_pixel[0] < 0 or final_pixel[0] >= background_image.shape[0] or final_pixel[1] < 0 or final_pixel[1] >= background_image.shape[1]:
        return np.array([1.0, 0.0, 1.0])
    
    return background_image[final_pixel]





def main(): 
    # load image
    img = mpimg.imread('image.jpg')
    
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    
    height, width = img.shape[:2]
    
    # output image 
    lensed_image = np.zeros_like(img)
    
    # obesrver properties
    r_obs = 100.0 * M
    vertical_fov_deg = 40.0
    
    vertical_fov: float = np.radians(vertical_fov_deg)
    horizontal_fov = 2 * np.arctan(np.tan(vertical_fov / 2) * width / height)
    fov = (horizontal_fov, vertical_fov)
    
    render_loop_around: bool = False # poor naming, I don't really know how to call this variable. If True, when the ray lands outside the bounds of the background image, it will be tiled instead of being marked as out of bounds.
    
    # alpha lookup table
    alpha_lookup: dict[tuple[int, int], float] = {}
    for y in range(height):
        for x in range(width):
            alpha = round(pixel_to_angles((y, x), (height, width), fov)[0], 4)
            if alpha < viewing_angle_to_impact_parameter(B_CRIT, r_obs):
                continue
            
            alpha_lookup[(y, x)] = alpha
    
    unique_alpha = set(alpha_lookup.values())
    
    final_alpha_lookup: dict[float, float] = {}
    for alpha in tqdm(unique_alpha, desc="Precomputing alpha lookup", unit="alpha"):
        final_alpha_lookup[alpha] = get_final_ray_angle(r_obs, alpha)
    
    # render output image
    for y in tqdm(range(height), desc="Rendering", unit="row"):
        for x in range(width):
            alpha = alpha_lookup[(y, x)]
            if alpha < viewing_angle_to_impact_parameter(B_CRIT, r_obs):
                lensed_image[y, x] = np.array([0.0, 0.0, 0.0])
                continue
            
            final_alpha = final_alpha_lookup[alpha]
            theta = pixel_to_angles((y, x), (height, width), fov)[1]
            final_pixel = angles_to_pixel((final_alpha, theta), img.shape[:2], fov)
            
            # for debugging purposes, returns a specific color if the ray escapes but goes out of bounds of the background image
            if not render_loop_around:
                if final_pixel[0] < 0 or final_pixel[0] >= img.shape[0] or final_pixel[1] < 0 or final_pixel[1] >= img.shape[1]:
                    lensed_image[y, x] = np.array([1.0, 0.0, 1.0])
                    continue
            else:
                # Tile the background image when the ray lands outside bounds.
                final_pixel = (
                    final_pixel[0] % img.shape[0],
                    final_pixel[1] % img.shape[1],
                )
            
            color = img[final_pixel]
            lensed_image[y, x] = color
    
    
    """
    ! As I implement the alpha first look up table, this loop will become obsolete and will be replaced by a simple lookup from the precomputed table.
    for y in tqdm(range(height), desc="Rendering", unit="row"):
        for x in range(width):
            angles = pixel_to_angles((y, x), (height, width), fov)
            color = get_color_from_background(angles, img, fov, r_obs)
            lensed_image[y, x] = color
    """
    
    
    # save output image
    mpimg.imsave('lensed_image.png', lensed_image)

if __name__ == "__main__":
    main()
