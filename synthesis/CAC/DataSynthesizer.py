
import cv2
import numpy as np
import os
import random
import json
from scipy.ndimage import gaussian_filter

class DataSynthesizer:
    def __init__(self, config_path="synthesis_config_final.json", output_dir="synthesis_final"):
        self.config = self._load_config(config_path)
        self.dapi_params = self.config["dapi_config"]
        self.channel_params = self.config["channel_params"]
        self.bg_profiles = self.config["background_profiles"]
        self.genetics = self.config.get("genetics", {})
        
        self.output_dir = output_dir
        self.target_size = 108
        self.super_sampling = 2
        self.work_size = self.target_size * self.super_sampling

    def _load_config(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    # --- Helpers ---
    def _fractal_noise(self, shape, scale_factor=1.0, scales=[4.0, 2.0, 1.0], weights=[0.5, 0.3, 0.2]):
        noise = np.zeros(shape, dtype=np.float32)
        for s, w in zip(scales, weights):
                layer = gaussian_filter(np.random.normal(0, 1, shape), sigma=s*scale_factor)
                noise += layer * w
        return (noise - np.mean(noise)) / (np.std(noise) + 1e-6)

    def _draw_rotated_gaussian(self, img, cx, cy, sigma_x, sigma_y, theta_deg, amplitude):
        rows, cols = img.shape
        max_sigma = max(sigma_x, sigma_y)
        kernel_radius = int(4 * max_sigma)
        
        x1, x2 = int(cx - kernel_radius), int(cx + kernel_radius + 1)
        y1, y2 = int(cy - kernel_radius), int(cy + kernel_radius + 1)
        
        x1 = max(0, x1); x2 = min(cols, x2)
        y1 = max(0, y1); y2 = min(rows, y2)
        
        if x1 >= x2 or y1 >= y2: return
        
        y_grid, x_grid = np.mgrid[y1:y2, x1:x2]
        dx = x_grid - cx
        dy = y_grid - cy
        
        theta = np.deg2rad(theta_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        dx_rot = dx * cos_t + dy * sin_t
        dy_rot = -dx * sin_t + dy * cos_t
        
        sx2 = 2 * sigma_x**2 + 1e-6
        sy2 = 2 * sigma_y**2 + 1e-6
        g = np.exp(-(dx_rot**2 / sx2 + dy_rot**2 / sy2))
        
        img[y1:y2, x1:x2] += g * amplitude

    def _draw_irregular_spot(self, img, cx, cy, a_radius, b_radius, angle, base_amplitude, structure='simple'):
        sub_spots = 1
        spread_factor = 0.1

        if structure == 'doublet':
            sub_spots = random.choice([2, 2, 3])
            spread_factor = 0.6 
        elif structure == 'cluster':
            sub_spots = random.randint(2, 4)
            spread_factor = 0.4
        else:
            if random.random() < 0.2:
                sub_spots = 2
                spread_factor = 0.25

        for i in range(sub_spots):
            if sub_spots == 1:
                size_frac = 1.0; amp_frac = 1.0
            else:
                is_main = (i == 0)
                size_frac = random.uniform(0.6, 0.9) if is_main else random.uniform(0.3, 0.7)
                amp_frac = 1.0 if is_main else random.uniform(0.5, 0.8)

            rad_angle = np.deg2rad(angle)
            u_off = random.gauss(0, a_radius * spread_factor)
            v_off = random.gauss(0, b_radius * spread_factor * 0.5)
            
            off_x = u_off * np.cos(rad_angle) - v_off * np.sin(rad_angle)
            off_y = u_off * np.sin(rad_angle) + v_off * np.cos(rad_angle)
            
            eff_sigma_x = (a_radius * size_frac) / 2.0
            eff_sigma_y = (b_radius * size_frac) / 2.0
            blob_angle = angle + random.uniform(-20, 20)
            blob_amp = base_amplitude * amp_frac * random.uniform(0.85, 1.15)
            
            self._draw_rotated_gaussian(img, cx + off_x, cy + off_y, eff_sigma_x, eff_sigma_y, blob_angle, blob_amp)

    # --- Core Generation ---
    def generate_nucleus(self, return_meta=False):
        scale = self.super_sampling
        rows, cols = self.work_size, self.work_size
        r_cfg = self.dapi_params["geometry"]
        
        # Geometry
        raw_radius = np.random.normal(r_cfg["radius_mean"] * scale, r_cfg["radius_std"] * scale)
        target_radius = np.clip(raw_radius, r_cfg["radius_min"] * scale, r_cfg["radius_max"] * scale)
        
        raw_ratio = np.random.normal(r_cfg["eccentricity_mean"], r_cfg["eccentricity_std"])
        ratio_ba = np.clip(raw_ratio, r_cfg["eccentricity_min"], r_cfg["eccentricity_max"])
        
        a = target_radius / np.sqrt(ratio_ba)
        b = target_radius * np.sqrt(ratio_ba)
        
        margin = r_cfg["margin_safety"] * scale
        max_d = (rows / 2.0) - margin
        if a > max_d:
            scale_fix = max_d / a
            a *= scale_fix
            b *= scale_fix

        cx, cy = rows // 2, cols // 2
        angle = random.uniform(0, 180)
        
        shape_meta = {"radius_px": float(target_radius / scale), "eccentricity": float(ratio_ba), "angle": float(angle)}
        
        mask = np.zeros((rows, cols), dtype=np.uint8)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(a), int(b)), float(angle), 0, 360, 255, -1)
        
        # Deformation
        dx = np.zeros((rows, cols), dtype=np.float32)
        dy = np.zeros((rows, cols), dtype=np.float32)
        k_base = np.clip(np.random.normal(0.50, 0.05), 0.40, 0.65)
        dx += gaussian_filter(np.random.normal(0, 1, (rows, cols)), sigma=25*scale) * k_base * 15 * scale
        dy += gaussian_filter(np.random.normal(0, 1, (rows, cols)), sigma=25*scale) * k_base * 15 * scale
        
        if random.random() < 0.80: # Boundary mutations
            theta = random.uniform(0, 2 * np.pi)
            u_x = a * np.cos(theta); u_y = b * np.sin(theta)
            rad_angle = np.deg2rad(angle)
            rot_x = u_x * np.cos(rad_angle) - u_y * np.sin(rad_angle)
            rot_y = u_x * np.sin(rad_angle) + u_y * np.cos(rad_angle)
            mx, my = int(cx + rot_x), int(cy + rot_y)
            
            vec_x, vec_y = mx - cx, my - cy
            norm = np.sqrt(vec_x**2 + vec_y**2) + 1e-6
            dir_x, dir_y = vec_x/norm, vec_y/norm
            
            sigma_mut = random.uniform(5.0, 8.0) * scale
            str_mut = random.uniform(5.0, 10.0) * scale * (1 if random.random() < 0.5 else -1)
            
            ksize = int(sigma_mut * 4) | 1
            gk = cv2.getGaussianKernel(ksize, sigma_mut); gk = gk * gk.T 
            gk = gk / gk.max() * str_mut
            
            y1, y2 = max(0, my - ksize//2), min(rows, my + ksize//2 + 1)
            x1, x2 = max(0, mx - ksize//2), min(cols, mx + ksize//2 + 1)
            gy1, gy2 = max(0, -(my - ksize//2)), ksize - max(0, (my + ksize//2 + 1) - rows)
            gx1, gx2 = max(0, -(mx - ksize//2)), ksize - max(0, (mx + ksize//2 + 1) - cols)

            if (y2>y1) and (x2>x1):
                dx[y1:y2, x1:x2] += gk[gy1:gy2, gx1:gx2] * dir_x
                dy[y1:y2, x1:x2] += gk[gy1:gy2, gx1:gx2] * dir_y
        
        x_grid, y_grid = np.meshgrid(np.arange(cols), np.arange(rows))
        map_x = (x_grid + dx).astype(np.float32)
        map_y = (y_grid + dy).astype(np.float32)
        mask_warped = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_LINEAR)
        _, mask_warped = cv2.threshold(mask_warped, 127, 255, cv2.THRESH_BINARY)
        
        # Intensity
        dist_map = cv2.distanceTransform(mask_warped, cv2.DIST_L2, 5)
        dist_map = gaussian_filter(dist_map, sigma=scale*2.0)
        norm_dist = dist_map / (np.max(dist_map) if np.max(dist_map) > 0 else 1)
        
        i_cfg = self.dapi_params["intensity_profile"]
        t_cfg = self.dapi_params["biological_texture"]
        
        hat_power = np.clip(np.random.normal(i_cfg["hat_power_mean"], i_cfg["hat_power_std"]), i_cfg["hat_power_min"], i_cfg["hat_power_max"])
        hat = i_cfg["pedestal"] + (1.0 - i_cfg["pedestal"]) * np.power(norm_dist, hat_power)
        
        base_texture = self._fractal_noise((rows, cols), scale_factor=scale, scales=[5.0, 2.5, 1.0], weights=[0.6, 0.3, 0.1])
        clump_smooth = gaussian_filter((gaussian_filter(base_texture, sigma=1.0) > 0.8).astype(np.float32), sigma=t_cfg["blur_clump"]*scale)
        nucleoli_smooth = gaussian_filter((self._fractal_noise((rows, cols), scale_factor=scale, scales=[3.0, 1.5]) > 1.2).astype(np.float32), sigma=t_cfg["blur_nucleoli"]*scale)
        
        rim_mask = (norm_dist < 0.15).astype(np.float32) * (norm_dist > 0.01).astype(np.float32)
        rim_smooth = gaussian_filter(rim_mask * ((gaussian_filter(np.random.normal(0,1,(rows,cols)), sigma=t_cfg["blur_rim"]*scale)>0).astype(np.float32)*0.5+0.5), sigma=1.0*scale)
        
        texture_map = 1.0 + t_cfg["weight_base_chromatin"] * base_texture + t_cfg["weight_heterochromatin"] * clump_smooth + t_cfg["weight_nucleoli"] * nucleoli_smooth + t_cfg["weight_nuclear_rim"] * rim_smooth
        hat = hat * texture_map
        tex_scaled = gaussian_filter(np.random.normal(0, 1.0, (rows, cols)), sigma=1.0) * 0.15
        
        # Background
        bg_cfg = self.dapi_params["background"]
        bg_mean = max(bg_cfg["min_level"], np.random.normal(bg_cfg["mean_base"], bg_cfg["mean_std"]))
        bg_noise = np.full((rows, cols), bg_mean, dtype=np.float32) + gaussian_filter(np.random.normal(0,1,(rows,cols)), sigma=30*scale)*2.0
        
        g_cfg = self.dapi_params["global_intensity"]
        nuc_int_target = np.clip(np.random.normal(g_cfg["target_mean"], g_cfg["target_std"]), g_cfg["clip_min"], g_cfg["clip_max"])
        
        hat_masked = hat[mask_warped > 0]
        hat_mean = np.mean(hat_masked) if hat_masked.size > 0 else 1.0
        amplitude = np.clip((nuc_int_target - bg_mean) / hat_mean, 1, 255)
        
        final_high = bg_noise.copy()
        np.putmask(final_high, mask_warped > 0, amplitude * hat + tex_scaled)
        
        psf_sigma = random.uniform(2.2, 3.8) * scale
        final_high = gaussian_filter(final_high, sigma=psf_sigma)
        
        final_img = cv2.resize(final_high, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        mask_final = cv2.resize(mask_warped, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        
        read_std = np.random.uniform(0.7, 1.0)
        final_img = np.clip(final_img + np.random.normal(0, read_std, final_img.shape), 0, 255).astype(np.uint8)
        
        if return_meta:
            shape_meta.update({"target_mean_intensity": float(nuc_int_target), "background_mean": float(bg_mean), "psf_sigma": float(psf_sigma / scale)})
            return final_img, mask_final, shape_meta
        return final_img, mask_final

    def generate_spot_channel(self, mask_dapi, channel_name, force_count=None, occupied_mask=None, channel_meta_override=None):
        params = self.channel_params[channel_name]
        scale = self.super_sampling
        rows, cols = mask_dapi.shape
        work_rows, work_cols = rows * scale, cols * scale
        
        bg_profile = self.bg_profiles[channel_name]
        bg_out_target = bg_profile['out_mean']
        bg_nuc_target = bg_profile['nuc_mean']
        
        channel_img = np.full((work_rows, work_cols), bg_out_target, dtype=np.float32)
        
        mask_highres = cv2.resize(mask_dapi, (work_cols, work_rows), interpolation=cv2.INTER_NEAREST)
        mask_bin = (mask_highres > 0).astype(np.uint8)
        
        # Soft mask
        dist_in = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 5)
        dist_out = cv2.distanceTransform(1 - mask_bin, cv2.DIST_L2, 5)
        nuc_bg_mask = 1.0 / (1.0 + np.exp(-(dist_in - dist_out) / (3.0 * scale + 1e-6)))
        
        nucleus_delta = max(1.0, (bg_nuc_target - bg_out_target) * 0.35 + np.random.normal(0, 1.0))
        channel_img += nuc_bg_mask * nucleus_delta
        
        # # Bio Noise (smoothed)
        # amp = params.get("bio_noise_amp", 8.0)
        # # Use a smoother base fractal noise to avoid sharp "holes"
        # raw_noise = self._fractal_noise((work_rows, work_cols), scale_factor=scale, scales=[8.0, 4.0], weights=[0.7, 0.3])
         
        # # Fix: Clamp negative noise values to prevent "black holes/dots"
        # # Deep negative values in fractal noise cause exp() to drop to ~0, creating unbiological black voids.
        # # Clamping at -1.5 preserves texture variation while filling the deepest holes.
        # raw_noise = np.maximum(raw_noise, 4.0)

        # bio_noise = amp * np.exp(raw_noise * 0.4) 
        # # Stronger smoothing to simulate fluid background haze
        # bio_noise = gaussian_filter(bio_noise, sigma=2.0*scale)
        # channel_img += nuc_bg_mask * bio_noise
        
        # Dark Stars
        ys_nuc, xs_nuc = np.where(mask_highres > 0)
        if len(xs_nuc) > 0:
            count_min = params.get("dark_star_count_min", 15)
            count_max = params.get("dark_star_count_max", 45)
            num_stars = random.randint(count_min, count_max)
            min_amp = 12.0 if channel_name == 'Aqua' else 20.0
            max_amp = 30.0 if channel_name == 'Aqua' else 50.0
            
            for _ in range(num_stars):
                idx = random.randint(0, len(xs_nuc) - 1)
                sx, sy = xs_nuc[idx], ys_nuc[idx]
                sigma_long = random.uniform(1.2, 3.0) * scale
                amplitude = random.uniform(min_amp, max_amp)
                tmp = np.zeros_like(channel_img)
                self._draw_rotated_gaussian(tmp, sx, sy, sigma_long, sigma_long * random.uniform(0.45, 0.85), random.uniform(0, 180), amplitude)
                channel_img += tmp * nuc_bg_mask

        # Spots
        num_spots = force_count if force_count is not None else params['count']
        current_occupied = np.zeros((work_rows, work_cols), dtype=bool) if occupied_mask is None else cv2.resize(occupied_mask.astype(np.uint8), (work_cols, work_rows), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        kernel = np.ones((int(8 * scale), int(8 * scale)), np.uint8)
        valid_region = cv2.erode(mask_highres, kernel, iterations=1)
        ys_valid, xs_valid = np.where(valid_region > 0)
        
        spots_meta = []
        spot_mask = np.zeros((work_rows, work_cols), dtype=np.uint8)
        
        ratio = np.sqrt(1 - params['ecc']**2)
        target_area_scaled = params['area'] * (scale**2)
        base_a = np.sqrt(target_area_scaled / (np.pi * ratio))
        base_b = base_a * ratio

        # Determine if we need to boost this channel to 3-5 spots? Handled by caller via force_count
        
        if len(xs_valid) > 0:
            for _ in range(num_spots):
                retry = 0
                while retry < 50:
                    idx = random.randint(0, len(xs_valid) - 1)
                    cx, cy = xs_valid[idx], ys_valid[idx]
                    
                    # Spot exclusion
                    overlap = False
                    r_ex = int(base_a * 1.5)
                    y_min, y_max = max(0, cy - r_ex), min(work_rows, cy + r_ex)
                    x_min, x_max = max(0, cx - r_ex), min(work_cols, cx + r_ex)
                    if np.any(current_occupied[y_min:y_max, x_min:x_max]):
                        retry += 1
                        continue
                        
                    # Draw
                    angle = random.uniform(0, 180)
                    temp_layer = np.zeros((work_rows, work_cols), dtype=np.float32)
                    self._draw_irregular_spot(temp_layer, cx, cy, base_a, base_b, angle, 1000.0, structure=params.get('structure', 'simple'))
                    
                    # Ripples
                    if random.random() < 0.30:
                        for r_idx in range(random.randint(1, 4)):
                            dist_mult = 1.7 + 0.65*r_idx
                            cv2.ellipse(temp_layer, (cx, cy), (int(base_a*dist_mult), int(base_b*dist_mult)), angle, 0, 360, (35.0*(0.55**r_idx),), max(1, int(0.9*scale)))

                    blur_sigma = 0.8 * scale
                    spot_blob = gaussian_filter(temp_layer, sigma=blur_sigma)
                    
                    cur_max = np.max(spot_blob)
                    if cur_max > 0:
                        target_max = max(np.random.normal(params['int_max'], 45.0), bg_out_target + 40 + bg_nuc_target) - (bg_out_target + bg_nuc_target)
                        spot_blob = spot_blob * (target_max / cur_max)
                        channel_img += spot_blob
                        
                        spots_meta.append({"x": float(cx)/scale, "y": float(cy)/scale, "intensity": float(target_max + bg_out_target + bg_nuc_target)})
                        
                        # Update Occupied
                        _, sm = cv2.threshold(temp_layer, 50.0, 255, cv2.THRESH_BINARY)
                        spot_mask = cv2.bitwise_or(spot_mask, sm.astype(np.uint8))
                        cv2.ellipse(current_occupied.view(np.uint8), (int(cx), int(cy)), (int(base_a*1.2), int(base_b*1.2)), angle, 0, 360, (1,), -1)
                        break
                    retry += 1
                    
        # PSF & Fog
        psf_sigma = random.uniform(1.2, 1.8) * (scale / 2.0)
        final_img = gaussian_filter(channel_img, sigma=psf_sigma)
        
        clouds = gaussian_filter(np.random.normal(0, 1.0, final_img.shape), sigma=15.0*scale); clouds /= (np.std(clouds) + 1e-6)
        fog = 1.5 * clouds
        outside_field = bg_out_target + fog
        
        final_img = outside_field * (1.0 - nuc_bg_mask) + final_img * nuc_bg_mask
        
        final_img = cv2.resize(final_img, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        
        # Shot Noise - Disabled to prevent black artifacts in background
        # nuc_bg_mask_target = cv2.resize(nuc_bg_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        # shot_noise = np.random.normal(0, 1.0, final_img.shape) * np.sqrt(np.maximum(final_img, 0)) * nuc_bg_mask_target
         
        final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        #final_img = np.clip(final_img, 0, 255).astype(np.uint8)
        mask_final = cv2.resize(spot_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        _, mask_final = cv2.threshold(mask_final, 127, 255, cv2.THRESH_BINARY)
        mask_occupied_ret = cv2.resize(current_occupied.astype(np.uint8), (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        meta = { "spots": spots_meta }
        return final_img, mask_final, mask_occupied_ret, meta

    def get_spot_counts(self, cls_name):
        counts = {'Aqua': 2, 'Gold': 2, 'Green': 2, 'Red': 2}
        if cls_name == 'Normal':
            return counts
        
        elif cls_name == 'Deletion':
            target_channel = random.choice(['Aqua', 'Gold', 'Green', 'Red'])
            counts[target_channel] = 1
            return counts
            
        elif cls_name == 'Gain':
            # Gain: Strictly ONE channel amplification
            target = random.choice(['Aqua', 'Green', 'Red', 'Gold'])
            probs = self.genetics['Gain']['probs']
            vals = self.genetics['Gain']['values']
            counts[target] = random.choices(vals, weights=probs)[0]
            return counts
            
        elif cls_name == 'CAC':
            # CAC: At least 2 channels amplification
            # Common patterns: (Aqua+Gold) OR (Green+Red) due to chromosomal pairing
            probs = self.genetics['CAC']['probs']
            vals = self.genetics['CAC']['values']
            def get_val(): return random.choices(vals, weights=probs)[0]
            
            pattern = random.choices(['chr10', 'chr3', 'complex'], weights=[0.45, 0.45, 0.10])[0]
            
            if pattern == 'chr10': # Aqua (CEP10) + Gold (10q22.3)
                v = get_val()
                counts['Aqua'] = v; counts['Gold'] = v
            elif pattern == 'chr3': # Green (3q29) + Red (3p22.1)
                v = get_val()
                counts['Green'] = v; counts['Red'] = v
            else: # Complex (Both pairs)
                v1 = get_val()
                counts['Aqua'] = v1; counts['Gold'] = v1
                v2 = get_val()
                counts['Green'] = v2; counts['Red'] = v2
            
            return counts
            
        return counts

    def generate_full_sample(self, cls, sample_id, save_dir):
        # DAPI
        img_dapi, mask_dapi, dapi_meta = self.generate_nucleus(return_meta=True)
        cv2.imwrite(os.path.join(save_dir, f"{sample_id}_DAPI.png"), img_dapi)
        
        counts = self.get_spot_counts(cls)
        occupied_mask = None
        
        meta = {
            "class": cls, "id": sample_id, "spot_counts": counts, "dapi_params": dapi_meta, "channels": {}
        }
        
        channels_draw = ['Aqua', 'Gold', 'Green', 'Red']
        random.shuffle(channels_draw)
        
        for ch in channels_draw:
            img_ch, _, occupied_mask, ch_meta = self.generate_spot_channel(
                mask_dapi, ch,
                force_count=counts[ch],
                occupied_mask=occupied_mask
            )
            
            cv2.imwrite(os.path.join(save_dir, f"{sample_id}_{ch}.png"), img_ch)
            meta["channels"][ch] = {"count": counts[ch], "realized": ch_meta}
            
        with open(os.path.join(save_dir, f"{sample_id}_params.json"), 'w') as f:
            json.dump(meta, f, indent=2)
