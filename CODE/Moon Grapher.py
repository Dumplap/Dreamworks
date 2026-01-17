import numpy as np
import rasterio
from pyproj import Transformer
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta

# ==========================================
# 1. DATABASES & CONFIGURATION
# ==========================================
EQUATION_DB = {
    "CRESTED BUTTE":    [-0.00019532, 0.17832599, 253.03, -0.00004940, -0.18440000, 28.54],
    "NEDERLAND":        [-0.00029902, 0.19265837, 252.72,  0.00005877, -0.18878835, 27.22],
    "RMNP":             [-0.00030356, 0.19028541, 253.11,  0.00006633, -0.19150009, 27.44],
    "FRISCO":           [-0.00030248, 0.18846945, 253.14,  0.00004905, -0.19244469, 27.98],
    "ASPEN":            [-0.00019532, 0.17832599, 253.03, -0.00004940, -0.18440000, 28.36],
    "HOLY CROSS":       [-0.00030248, 0.18846945, 253.14,  0.00004905, -0.19244469, 27.98],
    "INDIAN PEAKS":     [-0.00029902, 0.19265837, 252.72,  0.00005877, -0.18878835, 27.22],
    "SAWATCH":          [-0.00024890, 0.18339772, 253.09, -0.00000018, -0.18842235, 28.26],
    "BERTHOD":          [-0.00029902, 0.19265837, 252.72,  0.00005877, -0.18878835, 27.22]
}

MOON_RADIUS = 0.25  
START_TIME = datetime.strptime("04:05:00", "%H:%M:%S")

def get_moon_pos(site_key, t_min):
    c = EQUATION_DB[site_key]
    az = c[0]*(t_min**2) + c[1]*t_min + (c[2])
    alt = c[3]*(t_min**2) + c[4]*t_min + c[5]
    return az, alt

def format_eclipse_time(t_min):
    actual_time = START_TIME + timedelta(minutes=t_min)
    return actual_time.strftime("%H:%M:%S")

def find_best_site(target_lat, target_lon):
    best_dist, best_site = float('inf'), "BERTHOD"
    results_path = "RESULTS"
    if not os.path.exists(results_path): return best_site, 0.0
    for root, dirs, files in os.walk(results_path):
        for file in files:
            if "TOTALITY" in file.upper() and "GPS" in file.upper():
                try:
                    df = pd.read_csv(os.path.join(root, file))
                    for _, row in df.iterrows():
                        d = math.sqrt((target_lat-row['Photo_Lat'])**2 + (target_lon-row['Photo_Lon'])**2)
                        if d < best_dist:
                            best_dist, best_site = d, os.path.basename(root).upper()
                except: continue
    return best_site, best_dist

def generate_bw_horizon(vrt_path, lat, lon, az_min, az_max):
    with rasterio.open(vrt_path) as src:
        dem = src.read(1)
        res = src.res[0] 
        to_crs = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        mx, my = to_crs.transform(lon, lat)
        px, py = ~src.transform * (mx, my)
        obs_elev = dem[int(py), int(px)] + 2.0
        az_range = np.linspace(az_min, az_max, 3000)
        altitudes = []
        for az in az_range:
            rad = math.radians(az)
            dr, dc = -math.cos(rad), math.sin(rad)
            max_s = -15.0
            for d in range(50, 25000, int(res)):
                tr, tc = int(py + dr*(d/res)), int(px + dc*(d/res))
                if 0 <= tr < src.height and 0 <= tc < src.width:
                    h = dem[tr, tc]
                    if h < -500: continue
                    s = math.degrees(math.atan2(h - obs_elev, d))
                    if s > max_s: max_s = s
                else: break
            altitudes.append(max_s)
    return az_range, np.array(altitudes)

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
coords_raw = input("Enter Lat/Lon: ")
try:
    t_lat = float(coords_raw.split(',')[0].strip())
    t_lon = float(coords_raw.split(',')[1].strip())
    
    site, _ = find_best_site(t_lat, t_lon)
    print(f"SITE: {site}")

    # --- NEW DIRECTORY LOGIC ---
    # Creates "Lunar Plots/SITE_NAME/"
    output_dir = os.path.join("Lunar Plots", site)
    os.makedirs(output_dir, exist_ok=True)
    # ---------------------------

    # 1. Preliminary Solve
    center_est = EQUATION_DB[site][2]
    az_wide, alt_wide = generate_bw_horizon(f"{site}/mosaic.vrt", t_lat, t_lon, center_est-20, center_est+20)
    
    t_intersect, t_disappear = None, None
    for t in np.linspace(0, 200, 10000):
        m_az, m_alt = get_moon_pos(site, t)
        idx = np.abs(az_wide - m_az).argmin()
        h_at_az = alt_wide[idx]
        if t_intersect is None and (m_alt - MOON_RADIUS) <= h_at_az: t_intersect = t
        if t_disappear is None and (m_alt + MOON_RADIUS) <= h_at_az:
            t_disappear = t
            break

    phases = {
        f"({format_eclipse_time(t_intersect - 5)})": t_intersect - 5.0,
        f"({format_eclipse_time(t_intersect)})": t_intersect,
        f"({format_eclipse_time(t_disappear)})": t_disappear
    }

    if t_intersect is not None:
        int_az, int_alt = get_moon_pos(site, t_intersect)
        print(f"\n--- INTERSECTION DATA ---")
        print(f"Time: {format_eclipse_time(t_intersect)}")
        print(f"Moon Position: Azimuth {int_az:.4f}°, Altitude {int_alt:.4f}°")
        print(f"--------------------------\n")
    else:
        print("Warning: No intersection found for this location.")

    # 2. Final Window Logic
    az_vals = [get_moon_pos(site, p)[0] for p in phases.values()]
    plot_min_az, plot_max_az = min(az_vals) - 2.5, max(az_vals) + 2.5
    az_axis, alt_axis = generate_bw_horizon(f"{site}/mosaic.vrt", t_lat, t_lon, plot_min_az, plot_max_az)

    # 3. Plotting & Aspect Ratio Fix
    fig, ax = plt.subplots(figsize=(18, 10), facecolor='black')
    ax.set_facecolor('black')
    
    plt.fill_between(az_axis, alt_axis, -20, color='white', zorder=1)
    
    for label, t_val in phases.items():
        m_az, m_alt = get_moon_pos(site, t_val)
        moon = Circle((m_az, m_alt), MOON_RADIUS, color='red', ec='white', lw=1.5, alpha=0.5, zorder=10)
        ax.add_patch(moon)
        plt.text(m_az, m_alt + 0.7, label, color='white', ha='center', fontweight='bold', fontsize=11)
    
    x_span = plot_max_az - plot_min_az
    y_min = np.min(alt_axis) - 0.5
    y_max = y_min + x_span
    
    plt.xlim(plot_min_az, plot_max_az)
    plt.ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')

    ax.tick_params(colors='white', labelsize=12)
    plt.title(f"{site} | {t_lat}, {t_lon}", color='white', fontsize=16, pad=20)
    
    # --- UPDATED EXPORT LOGIC ---
    filename = f"{t_lat:.4f} {t_lon:.4f}.png"
    save_path = os.path.join(output_dir, filename)
    
    plt.savefig(save_path, dpi=300, facecolor='black', bbox_inches='tight')
    plt.close(fig) # Close plot to free memory
    print(f"--- SUCCESS: Profile saved to {save_path} ---")

except Exception as e:
    print(f"Error: {e}")