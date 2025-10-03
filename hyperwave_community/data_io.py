"""GDS export and visualization utilities for photonic device fabrication.

This module provides functions for converting density arrays from topology
optimization into GDS II format files suitable for semiconductor fabrication.
It handles complex polygons with holes using hierarchical containment analysis
and includes utilities for visualizing results.

Main functions:
    generate_gds_from_density: Convert 2D density array to GDS file
    view_gds: Visualize GDS file contents
    gds_to_theta: Convert GDS file to theta array
    component_to_theta: Convert gdsfactory component to theta array
"""

import numpy as np
import jax.numpy as jnp
import gdstk
from skimage import measure
import os
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple, Dict, Any


# =============================================================================
# Private Helper Functions for GDS Processing
# =============================================================================

def _winding_number(polygon, point):
    """Calculate winding number for point-in-polygon test.

    The winding number algorithm determines if a point is inside a polygon
    by counting how many times the polygon winds around the point.

    Args:
        polygon: Array of (x,y) coordinates defining the polygon.
        point: (x,y) coordinate to test.

    Returns:
        Integer winding number. 0 means point is outside, non-zero means inside.
    """
    wn = 0
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        if p1[1] <= point[1]:
            # Use 2D cross product calculation
            cross_product = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            if p2[1] > point[1] and cross_product > 0:
                wn += 1
        elif p2[1] <= point[1]:
            cross_product = (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
            if cross_product < 0:
                wn -= 1
    return wn


def _is_clockwise(polygon):
    """Determine if polygon vertices are in clockwise order.

    Uses the shoelace formula to calculate signed area. In GDS context,
    clockwise polygons represent material regions while counter-clockwise
    represent holes.

    Args:
        polygon: Array of (x,y) coordinates defining the polygon.

    Returns:
        True if vertices are in clockwise order, False otherwise.
    """
    return np.sum((polygon[1:, 0] - polygon[:-1, 0]) * (polygon[1:, 1] + polygon[:-1, 1])) > 0


def _build_containment_hierarchy(polygons):
    """Build containment hierarchy for nested polygons.

    Determines which polygons are contained within others, essential for
    correctly handling nested structures with holes.

    Args:
        polygons: List of polygon coordinate arrays.

    Returns:
        Tuple of (roots, hierarchy) where:
        - roots: List of indices of top-level (non-contained) polygons
        - hierarchy: Dict mapping polygon index to list of contained indices
    """
    hierarchy = {i: [] for i in range(len(polygons))}
    is_contained = [False] * len(polygons)

    for i in range(len(polygons)):
        for j in range(len(polygons)):
            if i == j:
                continue

            # Use winding number for robust point-in-polygon test
            if _winding_number(polygons[j], polygons[i][0]) != 0:
                # Check if j is the smallest polygon that contains i
                is_smallest_container = True
                for k in hierarchy[j]:
                    if _winding_number(polygons[k], polygons[i][0]) != 0:
                        is_smallest_container = False
                        break

                if is_smallest_container:
                    hierarchy[j].append(i)
                    is_contained[i] = True

    roots = [i for i, contained in enumerate(is_contained) if not contained]
    return roots, hierarchy


def _split_contours_at_artifacts_aggressive(contours):
    """Aggressively remove artifacts from contours for cleaner rectangles.

    Args:
        contours: List of contour arrays from skimage.measure.find_contours.

    Returns:
        List of cleaned contours with artifacts removed.
    """
    clean_contours = []

    for contour in contours:
        if len(contour) < 4:
            clean_contours.append(contour)
            continue

        segments = np.diff(contour, axis=0)
        lengths = np.sqrt(np.sum(segments**2, axis=1))

        if len(lengths) < 2:
            clean_contours.append(contour)
            continue

        median_length = np.median(lengths)
        max_length = np.max(lengths)

        # Flag if max is more than 2x the median
        if max_length > 2 * median_length and max_length > 5:
            artifact_idx = np.argmax(lengths)

            if artifact_idx == len(lengths) - 1:
                clean_contour = contour[:-1]
            else:
                clean_contour = np.vstack([
                    contour[:artifact_idx+1],
                    contour[artifact_idx+2:] if artifact_idx+2 < len(contour) else []
                ])

            # Ensure closure
            if len(clean_contour) > 0 and not np.allclose(clean_contour[0], clean_contour[-1]):
                clean_contour = np.vstack([clean_contour, clean_contour[0:1]])

            clean_contours.append(clean_contour)
        else:
            clean_contours.append(contour)

    return clean_contours


def _split_contours_at_artifacts(contours):
    """Remove artifact connections from contours.

    Removes artifact segments and closes gaps to form proper polygons.

    Args:
        contours: List of contour arrays from skimage.measure.find_contours.

    Returns:
        List of clean, closed contours.
    """
    clean_contours = []
    total_artifacts_removed = 0

    for contour_idx, contour in enumerate(contours):
        if len(contour) < 3:
            continue

        # Calculate segment lengths to identify artifacts
        segments = np.diff(contour, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))

        if len(segment_lengths) == 0:
            continue

        # Find abnormally long segments (likely artifacts)
        median_length = np.median(segment_lengths)
        artifact_threshold = median_length * 3  # More aggressive threshold

        # Find artifact segments
        artifact_indices = []
        for i, length in enumerate(segment_lengths):
            if length > artifact_threshold:
                artifact_indices.append(i)

        if not artifact_indices:
            # No artifacts found, keep the contour as-is
            clean_contours.append(contour)
        else:
            # Remove artifact segments and create clean closed contours
            # Create a list of valid point indices (excluding artifact endpoints)
            valid_points = []
            for i in range(len(contour)):
                # Skip points that are endpoints of artifact segments
                is_artifact_endpoint = False
                for artifact_idx in artifact_indices:
                    if i == artifact_idx or i == (artifact_idx + 1) % len(contour):
                        is_artifact_endpoint = True
                        break

                if not is_artifact_endpoint:
                    valid_points.append(contour[i])

            # If we have enough valid points, create a closed contour
            if len(valid_points) >= 3:
                # Convert to numpy array and ensure it's closed
                clean_contour = np.array(valid_points)

                # Close the contour if needed (connect last point to first)
                if not np.allclose(clean_contour[0], clean_contour[-1], atol=1e-6):
                    clean_contour = np.vstack([clean_contour, clean_contour[0:1]])

                clean_contours.append(clean_contour)
                total_artifacts_removed += len(artifact_indices)

    # Single summary print at the end
    if total_artifacts_removed > 0:
        print(f"Processed {len(contours)} contours: removed {total_artifacts_removed} artifacts")

    return clean_contours


# =============================================================================
# Main GDS Generation Function
# =============================================================================

def generate_gds_from_density(
    density_array: np.ndarray,
    level: float = 0.5,
    output_filename: str = "output.gds",
    add_border: bool = True,
    resolution: float = 0.05
) -> str:
    """Convert 2D density array to GDS II format file.

    This function converts topology optimization results into fabrication-ready
    GDS files. It extracts contours, handles nested polygons with holes, and
    performs boolean operations to create the final geometry.

    The process includes:
        1. Extracting contours at specified threshold level
        2. Cleaning artifacts from contour extraction
        3. Building hierarchy for nested polygons
        4. Applying boolean operations for holes
        5. Writing to GDS II format with proper scaling

    Args:
        density_array: 2D array of density values (0-1) from optimization.
            Values above 'level' are considered material. Note: density arrays
            are typically at 2x the resolution of the structure.
        level: Contour threshold (default 0.5). Material is density > level.
        output_filename: Name for output GDS file.
        add_border: If True, adds zero padding for closed contour extraction.
        resolution: Structure grid resolution in micrometers per pixel (default 0.05).
            This is the resolution of your STRUCTURE, not the density array.
            The function automatically applies 2x downsampling since density
            arrays are at 2x finer resolution than the structure.

    Returns:
        Absolute path to the generated GDS file.

    Note:
        The density array maintains its original orientation. Both numpy
        and GDS coordinate systems are treated consistently. Since density
        arrays are at 2x the resolution of the structure, the function
        internally uses resolution/2 for scaling to get correct physical dimensions.

    Example:
        >>> # If your structure has 0.05 um/pixel resolution
        >>> # Your density array will have 0.025 um/pixel resolution (2x finer)
        >>> gds_path = generate_gds_from_density(
        ...     density_array=density,
        ...     resolution=0.05,  # Pass structure resolution, not density resolution
        ...     output_filename="device.gds"
        ... )
    """
    # Since density arrays are at 2x the resolution of the structure,
    # we need to use half the resolution for correct physical scaling
    density_resolution = resolution / 2.0

    # Add a 1-pixel border of zeros to ensure proper contour extraction
    # This is purely algorithmic, not decorative
    if add_border:
        density_array = np.pad(density_array, pad_width=1, mode='constant', constant_values=0)

    # Find contours at the specified level using marching squares algorithm
    # Note: We don't flip vertically - both numpy and GDS use the same coordinate system
    raw_contours = measure.find_contours(density_array, level)

    # Try to detect if we have simple rectangles and clean them
    processed_contours = []
    for contour in raw_contours:
        # Get the bounding box
        min_row = np.min(contour[:, 0])
        max_row = np.max(contour[:, 0])
        min_col = np.min(contour[:, 1])
        max_col = np.max(contour[:, 1])

        # Check if this looks like a rectangle (most points are on the edges)
        on_edges = 0
        tolerance = 1.0
        for point in contour:
            if (abs(point[0] - min_row) < tolerance or
                abs(point[0] - max_row) < tolerance or
                abs(point[1] - min_col) < tolerance or
                abs(point[1] - max_col) < tolerance):
                on_edges += 1

        edge_ratio = on_edges / len(contour)

        # If most points are on edges, just create a clean rectangle
        if edge_ratio > 0.9 and len(contour) > 4:
            # Create a clean rectangle from the bounding box
            rect = np.array([
                [min_row, min_col],
                [min_row, max_col],
                [max_row, max_col],
                [max_row, min_col],
                [min_row, min_col]  # Close the rectangle
            ])
            processed_contours.append(rect)
        else:
            # Not a rectangle, try artifact removal
            processed = _split_contours_at_artifacts_aggressive([contour])
            processed_contours.extend(processed)

    # If no processing was done, use original contours
    if not processed_contours:
        processed_contours = raw_contours

    # Convert processed contours to gdstk.Polygon objects
    # Note: c[:, ::-1] swaps x and y coordinates to match GDS convention
    # If we added a border, we need to subtract 1 from coordinates to compensate
    # Apply density_resolution scaling to convert pixel coordinates to physical dimensions
    # GDS uses micrometers, and our density_resolution is already in micrometers
    if add_border:
        gds_polygons = [gdstk.Polygon((c[:, ::-1] - 1) * density_resolution) for c in processed_contours]
    else:
        gds_polygons = [gdstk.Polygon(c[:, ::-1] * density_resolution) for c in processed_contours]

    if not gds_polygons:
        # Create an empty library and cell and write it to GDS
        lib = gdstk.Library()
        lib.new_cell("EMPTY")
        lib.write_gds(output_filename)
        return os.path.abspath(output_filename)

    # Build containment hierarchy to understand polygon nesting
    roots, hierarchy = _build_containment_hierarchy(processed_contours)

    final_polygons = []

    # Process each root polygon and its children
    for root_idx in roots:
        # Start with the root polygon as material
        result_poly = [gds_polygons[root_idx]]

        # Process children recursively using boolean operations
        children_to_process = list(hierarchy[root_idx])
        while children_to_process:
            child_idx = children_to_process.pop(0)

            # Determine operation based on winding order:
            # - Counter-clockwise children are holes (subtract with 'not')
            # - Clockwise children are material islands (add with 'or')
            op = 'not' if not _is_clockwise(processed_contours[child_idx]) else 'or'

            # Perform boolean operation to add or subtract the child polygon
            result_poly = gdstk.boolean(result_poly, gds_polygons[child_idx], op)

            # Add grandchildren to the processing list for recursive handling
            children_to_process.extend(hierarchy[child_idx])

        final_polygons.extend(result_poly)

    # Create GDS library and cell structure
    lib = gdstk.Library()
    cell = lib.new_cell("TOP")  # "TOP" is the standard name for the main cell

    # Add all final polygons to the cell
    for poly in final_polygons:
        cell.add(poly)

    # Save the GDS file
    lib.write_gds(output_filename)

    return os.path.abspath(output_filename)


# =============================================================================
# Visualization Function
# =============================================================================

def view_gds(gds_filepath: str, density_array: np.ndarray = None, figsize: tuple = (12, 6)):
    """Visualize GDS file contents with optional density comparison.

    Reads a GDS file and plots the polygons it contains. If the original
    density array is provided, displays both side-by-side for comparison.
    The visualization shows the full domain including cladding regions.

    Args:
        gds_filepath: Path to the GDS file to visualize.
        density_array: Optional original density array for comparison.
        figsize: Figure size as (width, height) tuple.

    Returns:
        Matplotlib figure object containing the visualization.

    Note:
        Polygons are displayed with semi-transparent blue fill and no edge
        outline. The full domain is shown including cladding regions when
        density array is provided. The density array uses grayscale colormap.
    """
    # Read the GDS file
    lib_verify = gdstk.read_gds(gds_filepath)
    cell_verify = lib_verify.top_level()[0]

    # Get polygons from the cell
    polygons = cell_verify.get_polygons()

    print(f"GDS file: {gds_filepath}")
    print(f"Number of polygons in GDS: {len(polygons)}")

    # Create figure with subplots if density is provided
    if density_array is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot original density array
        im = ax1.imshow(density_array, cmap='gray', origin='upper',
                        extent=[0, density_array.shape[1], 0, density_array.shape[0]])
        ax1.set_title(f'Original Density ({density_array.shape[0]}×{density_array.shape[1]} pixels)')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax1, label='Density')

        ax = ax2
    else:
        fig, ax = plt.subplots(figsize=(figsize[0]/2, figsize[1]))

    # Plot the GDS polygons
    for poly in polygons:
        # gdstk.Polygon objects have a .points attribute
        points = poly.points
        poly_patch = plt.Polygon(points, alpha=0.7, edgecolor='none',
                                 facecolor='blue', linewidth=0)
        ax.add_patch(poly_patch)

    # Set axis limits to show full domain
    # Always use density array dimensions if provided to show full domain including cladding
    if density_array is not None:
        # Show the full domain extent
        ax.set_xlim(0, density_array.shape[1])
        ax.set_ylim(0, density_array.shape[0])
    else:
        # If no density array, try to infer reasonable bounds from polygons
        if polygons:
            all_x = []
            all_y = []
            for poly in polygons:
                points = poly.points
                all_x.extend(points[:, 0])
                all_y.extend(points[:, 1])

            # Use actual polygon bounds with 1-pixel margin
            if all_x and all_y:
                # Fixed 1-pixel margin, not percentage-based
                margin = 1
                ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        else:
            # Default view if no reference
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)

    ax.set_aspect('equal')
    ax.set_xlabel('X (GDS units)')
    ax.set_ylabel('Y (GDS units)')
    ax.set_title(f'GDS Polygons ({len(polygons)} polygons)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# GDS to Theta Array Conversion
# =============================================================================

def gds_to_theta(
    gds_filepath: str,
    resolution: float = 0.05,
    layer: Optional[Union[int, Tuple[int, int]]] = None,
    waveguide_value: float = 1.0,
    background_value: float = 0.0
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Convert GDS file to theta array for photonic simulations.

    Reads a GDS II format file and converts it to a 2D density array suitable
    for use in photonic simulations. The function rasterizes polygons from the
    GDS file into a pixelated array at the specified resolution.

    Args:
        gds_filepath: Path to the GDS file to convert.
        resolution: Grid resolution in micrometers per pixel. Default 0.05 (50nm).
            Smaller values give higher resolution but larger arrays.
        layer: Specific layer to extract. If None, uses most common layer.
            Can be integer or tuple (layer, datatype).
        waveguide_value: Value for waveguide regions. Default 1.0.
        background_value: Value for background regions. Default 0.0.

    Returns:
        Tuple of (theta, info) where:
        - theta: JAX array with shape (ny, nx) containing waveguide and
            background values.
        - info: Dictionary with metadata including 'gds_file', 'cell_name',
            'shape', 'resolution', 'physical_size_um' (width, height),
            'bounding_box_um' (x_min, y_min, x_max, y_max), 'layer',
            'num_polygons', 'waveguide_value', 'background_value'.

    Raises:
        ValueError: If no top-level cells or polygons found in GDS file.
        ValueError: If specified layer not found in GDS file.

    Note:
        For large GDS files or fine resolution, this can be memory intensive.
        The rasterization uses matplotlib.path for polygon filling which is
        accurate but can be slow for complex polygons.
    """
    from matplotlib.path import Path

    # Convert resolution from nm to μm for internal calculations
    # Resolution is already in micrometers
    resolution_um = resolution

    # Read the GDS file
    library = gdstk.read_gds(gds_filepath)

    # Get the top cell (usually there's only one main cell)
    top_cells = library.top_level()
    if not top_cells:
        raise ValueError(f"No top-level cells found in {gds_filepath}")

    cell = top_cells[0]  # Use the first top-level cell

    # Get all polygons from the cell
    # gdstk.Cell.get_polygons() returns just a list of polygons
    polygons = cell.get_polygons()

    if not polygons:
        raise ValueError(f"No polygons found in {gds_filepath}")

    # If we want layer information, we need to check each polygon's layer
    if layer is not None:
        # Filter polygons by layer
        filtered_polygons = []
        for poly in polygons:
            # In gdstk, polygons have a layer attribute
            if hasattr(poly, 'layer') and poly.layer == layer:
                filtered_polygons.append(poly)

        if not filtered_polygons:
            # Get all unique layers for error message
            layers = set()
            for poly in polygons:
                if hasattr(poly, 'layer'):
                    layers.add(poly.layer)
            raise ValueError(f"Layer {layer} not found. Available layers: {sorted(layers)}")

        polygons = filtered_polygons
        used_layer = layer
    else:
        # Get the most common layer
        layer_counts = {}
        for poly in polygons:
            if hasattr(poly, 'layer'):
                layer_counts[poly.layer] = layer_counts.get(poly.layer, 0) + 1

        if layer_counts:
            used_layer = max(layer_counts, key=layer_counts.get)
        else:
            used_layer = 0

    # Get bounding box from all polygons
    all_points = []
    for poly in polygons:
        all_points.extend(poly.points)

    if not all_points:
        raise ValueError("No valid polygons found")

    all_points = np.array(all_points)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)

    # Calculate grid dimensions
    width = x_max - x_min
    height = y_max - y_min
    nx = int(np.ceil(width / resolution_um))
    ny = int(np.ceil(height / resolution_um))

    # Initialize theta array with background value
    theta_array = np.full((ny, nx), background_value, dtype=np.float32)

    # Rasterize each polygon
    for i, poly in enumerate(polygons):

        points = np.array(poly.points)

        if len(points) < 3:
            continue

        # Shift to origin and convert to pixels
        points_shifted = points - [x_min, y_min]
        points_pixels = points_shifted / resolution_um

        # Create path for point-in-polygon test
        path = Path(points_pixels)

        # Get bounding box in pixels for this polygon
        x_min_px = max(0, int(np.floor(points_pixels[:, 0].min())))
        x_max_px = min(nx, int(np.ceil(points_pixels[:, 0].max())))
        y_min_px = max(0, int(np.floor(points_pixels[:, 1].min())))
        y_max_px = min(ny, int(np.ceil(points_pixels[:, 1].max())))

        # Create a grid of points to test
        x_coords = np.arange(x_min_px, x_max_px) + 0.5
        y_coords = np.arange(y_min_px, y_max_px) + 0.5
        xx, yy = np.meshgrid(x_coords, y_coords)
        points_to_test = np.column_stack([xx.ravel(), yy.ravel()])

        # Test all points at once
        inside = path.contains_points(points_to_test)
        inside_mask = inside.reshape(len(y_coords), len(x_coords))

        # Update theta array
        theta_array[y_min_px:y_max_px, x_min_px:x_max_px][inside_mask] = waveguide_value

    # Convert to JAX array
    theta_jax = jnp.array(theta_array)

    # Print summary
    print(f"Extracted {len(polygons)} polygons from GDS file")

    # Prepare metadata
    info = {
        'gds_file': gds_filepath,
        'cell_name': cell.name,
        'shape': theta_jax.shape,
        'resolution': resolution,
        'physical_size_um': (width, height),
        'bounding_box_um': (x_min, y_min, x_max, y_max),
        'layer': used_layer,
        'num_polygons': len(polygons),
        'waveguide_value': waveguide_value,
        'background_value': background_value,
    }

    return theta_jax, info


def component_to_theta(
    component,
    resolution: float = 0.05,
    layer: Optional[Union[int, Tuple[int, int]]] = None,
    waveguide_value: float = 1.0,
    background_value: float = 0.0
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Convert gdsfactory component to theta array for hyperwave simulations.

    This function takes a gdsfactory component and converts it to a 2D theta array
    suitable for use in hyperwave simulations. It rasterizes the component geometry
    at the specified resolution.

    Args:
        component: A gf.Component instance to convert.
        resolution: Grid resolution in micrometers per pixel in the final STRUCTURE.
            Default is 0.05 (50nm). Note: theta array will be generated at 2x this
            resolution since theta is downsampled by 2 during structure creation.
        layer: Specific layer to extract. If None, uses first available layer.
            Can be an integer or tuple (layer, datatype).
        waveguide_value: Value to assign to waveguide regions (default 1.0).
        background_value: Value to assign to background regions (default 0.0).

    Returns:
        theta: 2D JAX array with shape (ny, nx) containing the theta values.
            Values are waveguide_value where component exists, background_value elsewhere.
            Note: This array will be at 2x finer resolution than the final structure.
        info: Dictionary containing metadata:
            - 'component_name': Name of the component
            - 'shape': Shape of the theta array
            - 'theta_resolution_um': Resolution of theta array in micrometers per pixel
            - 'structure_resolution_um': Resolution of final structure after downsampling
            - 'physical_size_um': Physical size (width, height) in micrometers
            - 'bounding_box_um': Bounding box coordinates in micrometers
            - 'layer': Layer that was extracted

    Examples:
        >>> import gdsfactory as gf
        >>> from hyperwave.data_io import component_to_theta
        >>>
        >>> # Create component first
        >>> coupler = gf.components.coupler(gap=0.236, length=20.0)
        >>> theta, info = component_to_theta(coupler, resolution=0.05)
        >>>
        >>> # Or with a ring resonator
        >>> ring = gf.components.ring(radius=10.0)
        >>> theta, info = component_to_theta(ring, resolution=0.1)

    Note:
        Requires gdsfactory to be installed: pip install gdsfactory
    """
    try:
        import gdsfactory as gf
    except ImportError:
        raise ImportError("gdsfactory is required for component_to_theta. Install with: pip install gdsfactory")

    from matplotlib.path import Path

    # Use the component directly
    comp = component
    comp_name = comp.name if hasattr(comp, 'name') else 'component'

    # Get polygons from component
    all_polygons = comp.get_polygons()

    if not all_polygons:
        raise ValueError("No polygons found in component")

    # Select layer
    if layer is not None:
        if layer in all_polygons:
            polygons = all_polygons[layer]
            used_layer = layer
        else:
            available_layers = list(all_polygons.keys())
            raise ValueError(f"Layer {layer} not found. Available layers: {available_layers}")
    else:
        # Use first available layer
        used_layer = list(all_polygons.keys())[0]
        polygons = all_polygons[used_layer]
        if len(all_polygons) > 1:
            print(f"Multiple layers found: {list(all_polygons.keys())}. Using layer {used_layer}")

    # Get bounding box
    bbox = comp.bbox()
    x_min, y_min = bbox.left, bbox.bottom
    x_max, y_max = bbox.right, bbox.top

    # Calculate grid dimensions
    # Use half the resolution for theta since it gets downsampled by 2 in structure creation
    theta_resolution = resolution / 2.0
    width = x_max - x_min
    height = y_max - y_min
    nx = int(np.ceil(width / theta_resolution))
    ny = int(np.ceil(height / theta_resolution))

    # Initialize theta array with background value
    theta_array = np.full((ny, nx), background_value, dtype=np.float32)

    # Rasterize each polygon
    for polygon_obj in polygons:
        # Extract points from polygon
        points = []
        for point in polygon_obj.each_point_hull():
            # Convert from database units (nm) to micrometers
            x_um = point.x / 1000.0
            y_um = point.y / 1000.0
            points.append([x_um, y_um])

        if len(points) < 3:
            continue

        points = np.array(points)

        # Convert to pixel coordinates
        points_shifted = points - [x_min, y_min]
        points_pixels = points_shifted / theta_resolution

        # Create path for point-in-polygon testing
        path = Path(points_pixels)

        # Get bounding box in pixel coordinates
        x_min_px = max(0, int(np.floor(points_pixels[:, 0].min())))
        x_max_px = min(nx, int(np.ceil(points_pixels[:, 0].max())))
        y_min_px = max(0, int(np.floor(points_pixels[:, 1].min())))
        y_max_px = min(ny, int(np.ceil(points_pixels[:, 1].max())))

        # Fill pixels inside polygon
        for y in range(y_min_px, y_max_px):
            for x in range(x_min_px, x_max_px):
                # Check if pixel center is inside polygon
                if path.contains_point([x + 0.5, y + 0.5]):
                    theta_array[y, x] = waveguide_value

    # Convert to JAX array
    theta_jax = jnp.array(theta_array)

    # Create info dictionary
    info = {
        'component_name': comp_name,
        'shape': theta_jax.shape,
        'theta_resolution_um': theta_resolution,
        'structure_resolution_um': resolution,
        'physical_size_um': (width, height),
        'bounding_box_um': (x_min, y_min, x_max, y_max),
        'layer': used_layer,
        'waveguide_value': waveguide_value,
        'background_value': background_value,
    }

    return theta_jax, info