import os
import glob
import argparse
import numpy as np
from pathlib import Path

# 1. Force Matplotlib to use a non-interactive backend (Essential for headless Linux servers)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Global Settings & Model Parameters ---
FS = 1  # Scaling factor
# Matplotlib parameters for formatting (similar to cmss10 and cmmi10)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.linewidth'] = 1.2 * FS

# Parameters from the MATLAB colorModel.chBfit 
# Rows: R, G, B | Cols: b1, b2, b3, b4, b5
CH_B_FIT = np.array([
    [145.7586,  134.7227,  130.9048],
    [-130.4334, -124.1164, -101.2598],
    [  0.4656,   -0.4106,   -0.4582],
    [  0.7014,    0.3524,    0.4840],
    [  2.4949,    2.2571,    5.9188]
])

def color_model(b, x):
    """
    Python equivalent of: @(b,x)real(b(1)+b(2)*(x-b(3))/b(4)./(1+((x-b(3))/b(4)).^b(5)).^(1/b(5)))
    We use complex arrays to avoid NaNs when raising negative numbers to fractional powers, 
    then extract the real part just like MATLAB.
    """
    x_c = x.astype(complex)
    term1 = (x_c - b[2]) / b[3]
    denom = (1 + term1**b[4])**(1 / b[4])
    result = b[0] + b[1] * term1 / denom
    return np.real(result)

def get_custom_colormap(psi_min, psi_max):
    psi_vals = np.linspace(psi_min, psi_max, 256)
    
    # Apply log transformation
    log10_psi = np.log10(psi_vals / 2.22)
    log10_psi[np.isinf(log10_psi)] = np.log10(0.001)
    
    # Calculate RGB channels
    R = color_model(CH_B_FIT[:, 0], log10_psi)
    G = color_model(CH_B_FIT[:, 1], log10_psi)
    B = color_model(CH_B_FIT[:, 2], log10_psi)
    
    # Clean, clip (0-255), and normalize (0-1)
    rgb = np.column_stack((R, G, B))
    rgb = np.clip(rgb, 0, 255) / 255.0
    
    return mcolors.ListedColormap(rgb)

# --- Main Processing Loop ---
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process simulation data and generate plots.")
    parser.add_argument("sim_dir", type=str, help="Name of the simulation directory (e.g., sim_20260219_133113)")
    args = parser.parse_args()

    root_dir = Path.cwd()
    
    # Find all 'psi*.dat' files in the target directory recursively using the provided argument
    search_pattern = str(root_dir / args.sim_dir / '**' / 'psi*.dat')
    all_files = glob.glob(search_pattern, recursive=True)
    
    if not all_files:
        print(f"No 'psi*.dat' files found in {args.sim_dir}. Please check the directory name.")
        return

    # Group files by their parent folder
    folders = {}
    for filepath in all_files:
        folder = Path(filepath).parent
        if folder not in folders:
            folders[folder] = []
        folders[folder].append(filepath)

    for folder, files in folders.items():
        # Sort files alphabetically to maintain time-step order
        files = sorted(files)
        if not files:
            continue
            
        # --- Data Import ---
        # Read the first file to get spatial dimensions
        psi_sample = np.loadtxt(files[0])
        psis = np.zeros((len(psi_sample), len(files)))
        
        for fileIdx, filePath in enumerate(files):
            psis[:, fileIdx] = np.loadtxt(filePath)
            
        # Convert to %
        plotZ = psis * 100
        
        # Meshgrid (T=cols, X=rows)
        T = np.linspace(0, 20, plotZ.shape[1])
        X = np.linspace(0, 6, plotZ.shape[0])
        plotT, plotX = np.meshgrid(T, X)
        
        # --- Plotting ---
        # Convert figure dimensions from cm to inches (1 inch = 2.54 cm)
        fig_width_in = (13.6 * FS) / 2.54
        fig_height_in = (5.42 * FS) / 2.54
        fig = plt.figure(figsize=(fig_width_in, fig_height_in), facecolor='white')
        
        # Determine data bounds and build colormap
        psi_max = np.max(plotZ)
        psi_min = max(0.01, np.min(plotZ))
        custom_cmap = get_custom_colormap(psi_min, psi_max)
        
        # Plot surface as 2D heatmap
        ax = fig.add_axes([0, 0, 1, 1]) 
        c = ax.pcolormesh(plotT, plotX, plotZ, cmap=custom_cmap, vmin=0, vmax=psi_max, shading='auto')
        
        # --- Figure Formatting ---
        ax.set_xlim([0, 20])
        ax.set_ylim([0, 6])
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_yticks([0, 3, 6])
        
        ax.set_xlabel(r'$t$ [min]', fontsize=11 * FS)
        ax.set_ylabel(r'$x$ [cm]', fontsize=11 * FS)
        
        # Precise Axis Positioning (Normalized figure coordinates 0.0 to 1.0)
        left = (1.1 * FS) / (13.6 * FS)
        bottom = (1.1 * FS) / (5.42 * FS)
        width = 8.6460 / 13.6
        height = 3.7238 / 5.42
        ax.set_position([left, bottom, width, height])
        
        # Colorbar Formatting and Positioning
        cbar_left = left + width + (0.2 / 13.6)
        cbar_width = (0.5 * FS) / (13.6 * FS)
        cbar_ax = fig.add_axes([cbar_left, bottom, cbar_width, height])
        
        cbar = fig.colorbar(c, cax=cbar_ax)
        cbar.set_ticks(np.linspace(0, np.floor(psi_max / 10) * 10, 3))
        cbar.ax.tick_params(labelsize=11 * FS)
        cbar.outline.set_linewidth(1.2 * FS)
        
        # --- Saving ---
        # Extract purely the directory name (e.g., ignores trailing slashes like 'sim_123/')
        sim_name = os.path.basename(os.path.normpath(args.sim_dir))
        
        # Save as SVG (vector quality)
        output_file = f"{sim_name}.svg"
        plt.savefig(output_file, format='svg')
        plt.close(fig) # Close figure to free memory
        
        print(f"Processed: {output_file}")

if __name__ == "__main__":
    main()