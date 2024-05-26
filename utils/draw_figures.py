import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from typing import Callable, Tuple

def draw_shape(shape_func: Callable[[Axes], None], filename: str, size_pixels: Tuple[int, int], *args, **kwargs) -> None:
    """
    Draw and save a shape using the provided shape drawing function.

    Args:
        shape_func (Callable[[Axes], None]): Function that draws the shape on the given Axes.
        filename (str): Name of the file where the image will be saved.
        size_pixels (Tuple[int, int]): Width and height of the image in pixels.
        *args: Additional positional arguments to pass to the shape drawing function.
        **kwargs: Additional keyword arguments to pass to the shape drawing function.
    """
    dpi = 300
    size_inches = (size_pixels[0] / dpi, size_pixels[1] / dpi)

    fig, ax = plt.subplots(figsize=size_inches, dpi=dpi)
    ax.set_aspect('equal')
    shape_func(ax, *args, **kwargs)
    plt.axis('off')
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.gca().set_facecolor('white')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def draw_triangle(ax: Axes) -> None:
    """
    Draw a triangle on the given Axes.

    Args:
        ax (Axes): The axes to draw the triangle on.
    """
    triangle = plt.Polygon([(0.5, 1), (0, 0), (1, 0)], edgecolor='black', facecolor='white', linewidth=3)
    ax.add_patch(triangle)

def draw_circle(ax: Axes) -> None:
    """
    Draw a circle on the given Axes.

    Args:
        ax (Axes): The axes to draw the circle on.
    """
    circle = plt.Circle((0.5, 0.5), 0.5, edgecolor='black', facecolor='white', linewidth=3)
    ax.add_patch(circle)

def draw_square(ax: Axes) -> None:
    """
    Draw a square on the given Axes.

    Args:
        ax (Axes): The axes to draw the square on.
    """
    square = plt.Rectangle((0, 0), 1, 1, edgecolor='black', facecolor='white', linewidth=3)
    ax.add_patch(square)

def draw_parallelogram(ax: Axes) -> None:
    """
    Draw a parallelogram on the given Axes.

    Args:
        ax (Axes): The axes to draw the parallelogram on.
    """
    parallelogram = plt.Polygon([(0.2, 0), (0.8, 0), (1, 1), (0.4, 1)], edgecolor='black', facecolor='white', linewidth=3)
    ax.add_patch(parallelogram)

def draw_rhombus(ax: Axes) -> None:
    """
    Draw a rhombus on the given Axes.

    Args:
        ax (Axes): The axes to draw the rhombus on.
    """
    rhombus = plt.Polygon([(0.5, 1), (1, 0.5), (0.5, 0), (0, 0.5)], edgecolor='black', facecolor='white', linewidth=3)
    ax.add_patch(rhombus)