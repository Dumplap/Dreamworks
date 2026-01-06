import numpy as np
import rasterio
import cv2
import math
import pyopencl as cl
import pandas as pd
from tqdm import tqdm
import os
from scipy.ndimage import gaussian_filter
import platform
from datetime import datetime, timedelta

# ==========================================
# 1. GLOBAL PARAMETERS & CONFIGURATION
# ==========================================
RUN_ALL_COMBINATIONS = False                 # Toggle: True = All Sites, False = Only the site below
SITE = "INDIAN PEAKS"
MODE = "TOTALITY"                           # Options: "TOTALITY" or "PENUMBRAL"

# Raycasting Variables (Meters)
MIN_DIST = 300.0
MAX_DIST = 1250.0
STEP_SIZE = 1.0
TOLERANCE = 0.25

# Site Equations: [az_quad, az_lin, az_const, inc_quad, inc_lin, inc_const]
EQUATION_DB = {
    "CRESTED BUTTE":    [-0.00019532, 0.17832599, 73.03, -0.00004940, -0.18440000, 28.54],
    "NEDERLAND":        [-0.00029902, 0.19265837, 72.72,  0.00005877, -0.18878835, 27.22],
    "RMNP":             [-0.00030356, 0.19028541, 73.11,  0.00006633, -0.19150009, 27.44],
    "FRISCO":           [-0.00030248, 0.18846945, 73.14,  0.00004905, -0.19244469, 27.98],
    "ASPEN":            [-0.00019532, 0.17832599, 73.03, -0.00004940, -0.18440000, 28.36],   # reuse crested butte
    "HOLY CROSS":        [-0.00030248, 0.18846945, 73.14,  0.00004905, -0.19244469, 27.98],   # reuse frisco
    "INDIAN PEAKS":        [-0.00029902, 0.19265837, 72.72,  0.00005877, -0.18878835, 27.22]   # reuse nederland
}

def get_params_at_time(x, site_key):
    c = EQUATION_DB[site_key]
    az_deg = c[0] * (x**2) + c[1] * x + c[2]
    inc_deg = c[3] * (x**2) + c[4] * x + c[5]
    return np.float32(az_deg % 360), np.float32(inc_deg)

# ==========================================
# 3. AUTOMATED RIDGE & BACKGROUND GENERATION
# ==========================================
def get_or_generate_ridges(dem, site_key, npy_path):
    if os.path.exists(npy_path):
        print(f"--- Loading cached ridges: {npy_path}")
        return np.load(npy_path)

    print(f"--- Generating new derivative ridges for {site_key}...")
    valid_mask = (dem > -9000)
    smoothed = gaussian_filter(dem, sigma=5)
    laplacian = cv2.Laplacian(smoothed, cv2.CV_32F, ksize=5)
    ridge_mask = (laplacian < -2) & valid_mask
    
    margin = 25
    ridge_mask[0:margin, :] = False; ridge_mask[-margin:, :] = False
    ridge_mask[:, 0:margin] = False; ridge_mask[:, -margin:] = False
    
    coords = np.argwhere(ridge_mask > 0).astype(np.int32)
    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
    np.save(npy_path, coords)
    
    return coords

def get_cached_background(dem, site_key, site_cache_dir, out_w, out_h):
    if not os.path.exists(site_cache_dir): 
        os.makedirs(site_cache_dir)
    
    # We add the resolution to the filename so we don't load the wrong size from cache
    cache_path = f"{site_cache_dir}/{site_key}_bg_{out_w}x{out_h}.npy"
    
    if os.path.exists(cache_path):
        print(f"--- Loading cached background: {cache_path}")
        # Optional: verify the shape of the cached file matches what we expect
        cached_img = np.load(cache_path)
        if cached_img.shape[0] == out_h and cached_img.shape[1] == out_w:
            return cached_img
    
    print(f"--- Generating new shaded relief layer: {out_w}x{out_h} ---")
    ls = np.gradient(dem)
    slope = np.pi/2. - np.arctan(np.sqrt(ls[0]**2 + ls[1]**2))
    aspect = np.arctan2(-ls[1], ls[0])
    shaded = (np.sin(np.radians(45))*np.sin(slope) + 
              np.cos(np.radians(45))*np.cos(slope)*np.cos(np.radians(315)-aspect))
    
    # Scale to the dynamic dimensions calculated in the GPU function
    img = cv2.resize(cv2.cvtColor((255*(shaded+1)/2).astype(np.uint8), cv2.COLOR_GRAY2BGR), (out_w, out_h))
    
    np.save(cache_path, img)
    return img

def format_offset_to_time(minutes_offset):
    # Baseline is 4:05 AM
    base_time = datetime.strptime("04:05", "%H:%M")
    actual_time = base_time + timedelta(minutes=int(minutes_offset))
    
    # Check operating system for the correct "no leading zero" flag
    fmt = "%#I:%M %p" if platform.system() == "Windows" else "%-I:%M %p"
    
    return actual_time.strftime(fmt)

def draw_text_with_shadow(img, text, pos, font, scale, color, thickness, base_size):
    # Calculate dynamic offset based on resolution
    offset = max(1, int(0.5 * base_size))
    
    # 1. Draw Shadow (Black)
    shadow_pos = (pos[0] + offset, pos[1] + offset)
    # Fainter shadow alternative
    cv2.putText(target_img, text, (pos[0] + shadow_offset, pos[1] + shadow_offset), font, font_scale, (100, 100, 100), int(thickness), cv2.LINE_AA)    
    # 2. Draw Main Text
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

# ==========================================
# 4. OPENCL KERNEL
# ==========================================
kernel_code = r"""
__kernel void trace_rays(
    __global const float* dem, __global const int* ridges,
    __global int* ground_hits, __global int* hit_found,
    const int rows, const int cols, const float dr, const float dc,
    const float tan_angle, const float min_dist, const float max_dist,
    const float step_size, const float tolerance, const float inc_deg
){
    int gid = get_global_id(0);
    int r0 = ridges[gid * 2]; int c0 = ridges[gid * 2 + 1];
    float h0 = dem[r0 * cols + c0];
    int blocked_f = 0, blocked_b = 0, has_match = 0;
    int hit_r = -1, hit_c = -1;

    for (float d = 1.0f; d < max_dist; d += step_size) {
        int tr = (int)(r0 + dr * d); int tc = (int)(c0 + dc * d);
        if (tr < 0 || tr >= rows || tc < 0 || tc >= cols) { blocked_f = 1; }
        else if (!blocked_f) {
            float h_ray = h0 - (d * tan_angle);
            if (dem[tr * cols + tc] > (h_ray + 0.5f)) { blocked_f = 1; }
            if (!blocked_f && d >= min_dist) {
                float ang = atan2(h0 - dem[tr * cols + tc], d) * 57.2957795f;
                if (ang <= inc_deg && ang > (inc_deg - tolerance)) {
                    has_match = 1; hit_r = tr; hit_c = tc;
                }
            }
        }
        int trb = (int)(r0 - dr * d), tcb = (int)(c0 - dc * d);
        if (trb >= 0 && trb < rows && tcb >= 0 && tcb < cols) {
            if (dem[trb * cols + tcb] > (h0 + d * tan_angle)) blocked_b = 1;
        }
        if (blocked_f && blocked_b) break;
    }
    if (has_match && !blocked_b) {
        ground_hits[gid * 2] = hit_r; ground_hits[gid * 2 + 1] = hit_c;
        hit_found[gid] = 1;
    } else { hit_found[gid] = 0; }
}
"""

# ==========================================
# 4. GPU ANALYSIS
# ==========================================

def run_gpu_analysis(current_site, current_mode):
    # Dynamic Constants
    if current_mode == "TOTALITY":
        T_S, T_E, T_O = 0, 57, 28.0
    else:
        T_S, T_E, T_O = -74, -20, -47.0

    folder_n = current_site.replace("_", " ").title()
    site_cache = f"cache/{folder_n}"
    vrt_p = f"{folder_n}/mosaic.vrt"
    npy_p = f"{site_cache}/ridges.npy"
    out_img = f"RESULTS/{folder_n} {current_mode}.png"
    out_csv = f"RESULTS/DATA/{folder_n} {current_mode}.csv"

    if not os.path.exists("RESULTS"): os.makedirs("RESULTS")

    print(f"--- PROCESSING: {folder_n} ({current_mode}) ---")
    
    # Initialize OpenCL
    print(f"--- Initializing GPU context...")
    platforms = cl.get_platforms()
    dev = next(d for p in platforms for d in p.get_devices(cl.device_type.GPU) if "GFX1102" in d.name.upper())
    ctx = cl.Context([dev]); queue = cl.CommandQueue(ctx)

    print(f"--- Loading DEM from {vrt_p}...")
    with rasterio.open(vrt_p) as src:
        dem = src.read(1).astype(np.float32)
        rows, cols = dem.shape

    # --- DYNAMIC RESOLUTION LOGIC ---
    # TARGET_RESOLUTION (e.g., 4000) defines the longest side
    scale = 4000 / max(rows, cols)
    out_w, out_h = int(cols * scale), int(rows * scale)

    ridge_coords = get_or_generate_ridges(dem, current_site, npy_p)[::5] 
    num_ridges = len(ridge_coords)
    
    # Ensure your get_cached_background accepts (dem, site_key, cache_dir, out_w, out_h)
    img = get_cached_background(dem, current_site, site_cache, out_w, out_h)

    # Set up GPU Buffers
    print(f"--- Transferring data to GPU buffers...")
    mf = cl.mem_flags
    dem_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=dem)
    ridge_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=ridge_coords.flatten())
    hit_buf = cl.Buffer(ctx, mf.WRITE_ONLY, ridge_coords.nbytes)
    flag_buf = cl.Buffer(ctx, mf.WRITE_ONLY, num_ridges * 4)
    prg = cl.Program(ctx, kernel_code).build()
    knl = cl.Kernel(prg, "trace_rays")

    results = []
    time_domain = np.arange(T_S, T_E + 1, 1)
    
    print(f"--- Starting Raycasting loop...")
    for x in tqdm(time_domain, desc="Raycasting", leave=False):
        az_deg, inc_deg = get_params_at_time(x, current_site)
        az_rad = math.radians(az_deg)
        dr, dc = np.float32(-math.cos(az_rad)), np.float32(math.sin(az_rad))
        knl.set_args(dem_buf, ridge_buf, hit_buf, flag_buf, np.int32(rows), np.int32(cols),
                     dr, dc, np.float32(math.tan(math.radians(inc_deg))),
                     np.float32(MIN_DIST), np.float32(MAX_DIST), np.float32(STEP_SIZE), 
                     np.float32(TOLERANCE), np.float32(inc_deg))
        cl.enqueue_nd_range_kernel(queue, knl, (num_ridges,), None)
        f, h = np.empty(num_ridges, dtype=np.int32), np.empty(num_ridges*2, dtype=np.int32)
        cl.enqueue_copy(queue, f, flag_buf); cl.enqueue_copy(queue, h, hit_buf)
        valid = np.where(f == 1)[0]
        for i in valid: results.append([x, ridge_coords[i,0], ridge_coords[i,1], h[i*2], h[i*2+1]])

    if not results or len(results) == 0:
        print(f"!!! No hits found for {current_site} {current_mode}")
        return 0, 0, 0.0  # Return floats to ensure consistency

    print(f"--- Hits found: {len(results)}. Mapping to visualization...")
    res_arr = np.array(results)
    times = res_arr[:, 0]
    ratios = (times - T_S) / (T_E - T_S)
    
    # 1. Distance Calculation (meters)
    # res_arr: [time, ridge_r, ridge_c, ground_r, ground_c]
    dist_meters = np.sqrt((res_arr[:, 1] - res_arr[:, 3])**2 + 
                          (res_arr[:, 2] - res_arr[:, 4])**2)

    # Scale coordinates for drawing
    g_rows = np.clip((res_arr[:, 3] * scale).astype(int), 2, out_h - 3)
    g_cols = np.clip((res_arr[:, 4] * scale).astype(int), 2, out_w - 3)
    r_rows = np.clip((res_arr[:, 1] * scale).astype(int), 2, out_h - 3)
    r_cols = np.clip((res_arr[:, 2] * scale).astype(int), 2, out_w - 3)

    # 2. Logic Filters
    dist_logic = (dist_meters >= 800.0) & (dist_meters <= 1225.0)
    
    if current_mode == "PENUMBRAL":
        time_logic = (times >= -55) & (times <= -40)
    else:
        time_logic = np.abs(times - T_O) <= 3

    mid_logic = time_logic & dist_logic

    # 3. Middle 50% Time Range (Washed-out logic)
    time_range = T_E - T_S
    center_50_logic = (times >= (T_S + time_range * 0.25)) & (times <= (T_E - time_range * 0.25))

    # 4. Visualization Mapping
    g_mask = np.zeros((out_h, out_w), dtype=np.uint8)
    
    # Apply layers: Outer Time (3) -> Mid 50% (1) -> Preferred (2)
    g_mask[g_rows, g_cols] = 3 
    g_mask[g_rows[center_50_logic], g_cols[center_50_logic]] = 1
    g_mask[g_rows[mid_logic], g_cols[mid_logic]] = 2   
    
    dilated = cv2.dilate(g_mask, np.ones((3,3), np.uint8))
    img[dilated == 1] = [0, 255, 0]        # Vibrant Green
    img[dilated == 2] = [255, 100, 255]    # Pink
    img[dilated == 3] = [100, 130, 100]    # Washed Green

    # Draw Skier Gradient on Ridges
    for idx in range(len(r_rows)):
        r, c, rat = r_rows[idx], r_cols[idx], ratios[idx]
        img[r:r+2, c:c+2] = [(255 * rat), 40, (255 * (1 - rat))]

    # --- 5. DYNAMIC, ANTI-ALIASED LEGEND WITH DROP SHADOWS ---
    base_size = out_w / 4000
    l_w, l_h = int(600 * base_size), int(420 * base_size)
    pad = int(40 * base_size)
    
    # Draw Background Semi-Transparent Box
    overlay = img.copy()
    cv2.rectangle(overlay, (out_w-l_w-pad, out_h-l_h-pad), (out_w-pad, out_h-pad), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    
    font = cv2.FONT_HERSHEY_DUPLEX
    tx = out_w - l_w - pad + int(30 * base_size)
    ty = out_h - l_h - pad + int(60 * base_size)
    l_step = int(55 * base_size)

    # Shadow Helper Logic
    def put_shadow_text(target_img, text, pos, font_scale, color, thickness):
        shadow_offset = max(1, int(2 * base_size))
        # Draw Black Shadow
        cv2.putText(target_img, text, (pos[0] + shadow_offset, pos[1] + shadow_offset), 
                    font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Draw Main Text
        cv2.putText(target_img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)

    # 1. Title
    put_shadow_text(img, f"{folder_n} ({current_mode})", (tx, ty), 1.0*base_size, (255,255,255), 2)
    
    # 2. Dynamic Time Formatting for Labels
    clock_start = format_offset_to_time(T_S)
    clock_end = format_offset_to_time(T_E)
    
    if current_mode == "PENUMBRAL":
        time_range_text = f"{format_offset_to_time(-55)} - {format_offset_to_time(-40)}"
    else:
        time_range_text = f"{format_offset_to_time(T_O-3)} - {format_offset_to_time(T_O+3)}"

    # 3. Legend Items (Status Markers)
    items = [
        ([255, 100, 255], f"Preferred: {time_range_text} & (800-1225m)"),
        ([0, 255, 0], "Photographer: Mid 50% Window"),
        ([100, 130, 100], "Photographer: Start/End")
    ]
    
    for color, text in items:
        ty += l_step
        # Smoother circles with a subtle dark outline
        cv2.circle(img, (tx + int(15*base_size), ty - int(10*base_size)), int(14*base_size), (20,20,20), -1, cv2.LINE_AA)
        cv2.circle(img, (tx + int(15*base_size), ty - int(10*base_size)), int(11*base_size), color, -1, cv2.LINE_AA)
        put_shadow_text(img, text, (tx + int(50*base_size), ty), 0.65*base_size, (255,255,255), 1)

    # 4. Skier Gradient Bar
    ty += int(l_step * 1.2)
    put_shadow_text(img, "Skier Time Gradient", (tx, ty), 0.7*base_size, (255,255,255), 1)
    
    ty += int(20 * base_size)
    grad_w = l_w - int(80 * base_size)
    bar_height = int(25 * base_size)
    
    # Draw the bar pixel by pixel
    for i in range(int(grad_w)):
        c_rat = i / grad_w
        bar_color = (int(255 * c_rat), 40, int(255 * (1 - c_rat)))
        cv2.line(img, (tx + i, ty), (tx + i, ty + bar_height), bar_color, 1)

    # 5. Dynamic Clock Labels on Bar Ends
    label_y = ty + bar_height + int(25 * base_size)
    # Start Time (Left)
    put_shadow_text(img, clock_start, (tx, label_y), 0.55*base_size, (255,255,255), 1)
    # End Time (Right-Aligned)
    text_size = cv2.getTextSize(clock_end, font, 0.55*base_size, 1)[0]
    put_shadow_text(img, clock_end, (tx + grad_w - text_size[0], label_y), 0.55*base_size, (255,255,255), 1)

    print(f"--- Outputting result to: {out_img}")
    cv2.imwrite(out_img, img)
    return len(res_arr), np.sum(mid_logic), (np.sum(mid_logic)/len(res_arr))*100

# ==========================================
# 4. MAIN
# ==========================================

if __name__ == "__main__":
    if not os.path.exists("RESULTS/DATA"): 
        os.makedirs("RESULTS/DATA")
        print("--- Created RESULTS/DATA directory ---")

    if RUN_ALL_COMBINATIONS:
        summary = []
        for s in EQUATION_DB.keys():
            for m in ["TOTALITY", "PENUMBRAL"]:
                try:
                    total, pink, perc = run_gpu_analysis(s, m)
                    summary.append({"Site": s, "Mode": m, "Pink": pink, "Percentage": round(perc, 2)})
                except Exception as e:
                    print(f"Error {s} {m}: {e}")
        
        df = pd.DataFrame(summary).sort_values(by="Percentage", ascending=False)
        df.to_csv("RESULTS/SUMMARY_COMPARISON.csv", index=False)
        print("\n--- BATCH SUMMARY ---\n", df)
        print(f"--- Summary CSV saved to RESULTS/SUMMARY_COMPARISON.csv ---")
    else:
        # Run just the single choice from the top
        run_gpu_analysis(SITE, MODE)