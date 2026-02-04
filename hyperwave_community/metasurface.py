"""Metasurface design and optimization utilities.

This module provides functions for creating and optimizing metasurface structures
with various geometric patterns and arrangements.
"""

import jax.numpy as jnp


def create_circle_array(size: int, radius: float) -> jnp.ndarray:
    """Create a single circle array.
    
    Args:
        size: Size of the square array
        radius: Radius of the circle
        
    Returns:
        Binary array with circle pattern
    """
    x = jnp.arange(0, size)
    y = jnp.arange(0, size)
    xx, yy = jnp.meshgrid(x, y, sparse=True)

    center_x = (size - 1) / 2
    center_y = (size - 1) / 2

    mask = (xx - center_x)**2 + (yy - center_y)**2 <= radius**2
    return jnp.where(mask, 1, 0)


def create_circle_grid(
    radius: float,
    edge_separation: float, 
    nx_circles: int,
    ny_circles: int,
    padding: int = 0
) -> jnp.ndarray:
    """Create a grid of circles with specified spacing and arrangement.
    
    Args:
        radius: Radius of each circle
        edge_separation: Edge-to-edge separation between adjacent circles
        nx_circles: Number of circles along x-axis
        ny_circles: Number of circles along y-axis  
        padding: Additional padding around the entire grid
        
    Returns:
        Binary array with grid of circles pattern
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if edge_separation < 0:
        raise ValueError(f"edge_separation must be non-negative, got {edge_separation}")
    if nx_circles < 1 or ny_circles < 1:
        raise ValueError(f"Number of circles must be >= 1, got nx={nx_circles}, ny={ny_circles}")
    if edge_separation == 0:
        print(f"Warning: edge_separation is 0, circles will be touching")
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")
    
    # Convert edge-to-edge separation to center-to-center separation
    center_separation = edge_separation + 2 * radius
    
    total_width = (nx_circles - 1) * center_separation + 2 * radius + 2 * padding
    total_height = (ny_circles - 1) * center_separation + 2 * radius + 2 * padding
    
    array_width = int(jnp.ceil(total_width))
    array_height = int(jnp.ceil(total_height))
    
    grid_array = jnp.zeros((array_width, array_height))
    
    start_x = padding + radius
    start_y = padding + radius
    
    x = jnp.arange(0, array_width)
    y = jnp.arange(0, array_height)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    
    for i in range(nx_circles):
        for j in range(ny_circles):
            center_x = start_x + i * center_separation
            center_y = start_y + j * center_separation
            
            circle_mask = (xx - center_x)**2 + (yy - center_y)**2 <= radius**2
            grid_array = jnp.where(circle_mask, 1, grid_array)
    
    return grid_array

 