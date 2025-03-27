"""
Plotly utility functions for visualization and charting needs.
"""
import plotly.colors as pc
import plotly.express as px
import numpy as np

def get_n_colors(n, palette='Plotly'):
    """
    Returns exactly `n` colors by extending a qualitative Plotly color palette.
    If n exceeds the palette size, colors will be recycled with slight variations.

    Parameters:
    - n: Number of colors to generate.
    - palette: Qualitative Plotly palette name (default: 'Plotly').
               Options include: 'Plotly', 'D3', 'G10', 'T10', 'Alphabet', etc.

    Returns:
    - List of HEX color codes from the specified qualitative palette.
    """
    # Get the base qualitative palette
    if palette == 'Rainbow':
        # For backward compatibility, use a continuous colorscale
        return pc.sample_colorscale('Rainbow', np.linspace(0, 1, n))
    
    # Get the requested qualitative palette
    palette_name = f"px.colors.qualitative.{palette}"
    try:
        base_colors = eval(palette_name)
    except (AttributeError, NameError):
        # Fallback to Plotly palette if the requested one doesn't exist
        base_colors = px.colors.qualitative.Plotly
    
    # If we need fewer colors than in the palette, return a subset
    if n <= len(base_colors):
        return base_colors[:n]
    
    # If we need more colors than in the palette, recycle with variations
    result_colors = []
    base_palette_size = len(base_colors)
    
    for i in range(n):
        # Get the base color from the palette (cycling through)
        base_color = base_colors[i % base_palette_size]
        
        # For the first cycle, use original colors
        if i < base_palette_size:
            result_colors.append(base_color)
        else:
            # For subsequent cycles, adjust brightness/saturation slightly
            # based on which cycle we're in
            cycle_num = i // base_palette_size
            
            # Convert hex to RGB for manipulation
            r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
            
            # Adjust brightness based on cycle number (alternate darker/lighter)
            factor = 0.8 if cycle_num % 2 == 1 else 1.2
            factor = min(factor, 1.5)  # Cap the brightening factor
            
            # Apply the adjustment and ensure values stay in valid range
            r = max(0, min(255, int(r * factor)))
            g = max(0, min(255, int(g * factor)))
            b = max(0, min(255, int(b * factor)))
            
            # Convert back to hex
            modified_color = f"#{r:02x}{g:02x}{b:02x}"
            result_colors.append(modified_color)
    
    return result_colors