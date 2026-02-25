
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

class ScratchSynthesizer:
    def __init__(self, canvas_size=(512, 512)):
        self.h, self.w = canvas_size
        
    def _get_bezier_points(self, p0, p1, p2, p3, num_points=200):
        """Standard Cubic Bezier Curve"""
        t = np.linspace(0, 1, num_points)
        # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        x = (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        return np.stack([x, y], axis=1)

    def _generate_perlin_1d(self, length, scale=10):
        """Simple 1D noise for roughness"""
        if length == 0: return np.zeros(0)
        # Interpolate random keypoints
        num_keypoints = max(2, length // scale)
        keypoints = np.random.randn(num_keypoints)
        x_keys = np.linspace(0, length, num_keypoints)
        x_interp = np.arange(length)
        noise = np.interp(x_interp, x_keys, keypoints)
        return noise

    def generate_scratch(self, start_pos=None, end_pos=None, width=3.0, intensity=0.8, roughness=0.5, waviness=0.0, param_bending=50):
        """
        Generate a scratch mask using Bezier curve + Gaussian profile + Noise.
        
        Args:
            width: Base width (sigma of Gaussian)
            intensity: Peak intensity (0.0 - 1.0)
            roughness: How much the edge/width deviates (0.0 - 1.0)
            waviness: Amplitude of high-freq sinusoidal/noise deviation to path (0.0 - 10.0+)
            param_bending: Control point offset magnitude (Curve arch height)
        """
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        
        # 1. Define Control Points
        if start_pos is None:
            start_pos = (random.randint(50, self.w-50), random.randint(50, self.h-50))
        if end_pos is None:
            angle = random.uniform(0, 2*np.pi)
            dist = random.uniform(100, 300)
            end_pos = (start_pos[0] + np.cos(angle)*dist, start_pos[1] + np.sin(angle)*dist)
            
        # Random control points for curvature
        # Use Perpendicular direction for "bending" to actually create an arc, 
        # rather than just random xy noise which might result in 'S' or flat lines.
        
        mid_x = (start_pos[0] + end_pos[0]) / 2
        mid_y = (start_pos[1] + end_pos[1]) / 2
        
        # Vector Start -> End
        v_x = end_pos[0] - start_pos[0]
        v_y = end_pos[1] - start_pos[1]
        dist = np.sqrt(v_x**2 + v_y**2)
        if dist == 0: dist = 1
        
        # Normal vector
        n_x = -v_y / dist
        n_y = v_x / dist
        
        # Determine bending direction (randomly left or right)
        bend_dir = random.choice([1, -1])
        
        # Control points pulled along normal
        # P1 and P2 slightly different to allow asymmetric arches
        offset1 = param_bending * random.uniform(0.8, 1.2) * bend_dir
        offset2 = param_bending * random.uniform(0.8, 1.2) * bend_dir
        
        # Also add small random jitter along the chord to prevent perfect parabolic look
        jitter = dist * 0.1
        
        p1 = (mid_x + n_x * offset1 + random.uniform(-jitter, jitter), 
              mid_y + n_y * offset1 + random.uniform(-jitter, jitter))
        p2 = (mid_x + n_x * offset2 + random.uniform(-jitter, jitter), 
              mid_y + n_y * offset2 + random.uniform(-jitter, jitter))
        
        # 2. Get Path Points
        path_length_est = np.linalg.norm(np.array(start_pos) - np.array(end_pos))
        num_steps = int(path_length_est * 2) # super sample steps
        points = self._get_bezier_points(start_pos, p1, p2, end_pos, num_steps)
        
        # Apply waviness (High frequency perturbation)
        if waviness > 0:
            # Generate perpendicular offsets
            # Use higher frequency noise than roughness
            wave_noise = self._generate_perlin_1d(num_steps, scale=5)
            
            # Apply to points (approximate normal direction application would be better, but simple xy addition works for chaos)
            # To be cleaner, let's just add it to X and Y independently
            wave_x = self._generate_perlin_1d(num_steps, scale=5)
            wave_y = self._generate_perlin_1d(num_steps, scale=5)
            
            points[:, 0] += wave_x * waviness
            points[:, 1] += wave_y * waviness

        # 3. Generate Noise for Width variation
        if roughness > 0.01:
            scale_val = int(10/(roughness+0.001))
            noise = self._generate_perlin_1d(num_steps, scale=scale_val)
            variance = width * 0.4 * roughness
            widths = width + noise * variance
        else:
            widths = np.full(num_steps, width)
            
        widths = np.clip(widths, 0.5, width * 2)
        
        # 4. Render 1D Gaussian Cross-section along path
        for i in range(len(points) - 1):
            pt = points[i]
            next_pt = points[i+1]
            
            # Tangent & Normal
            tangent = next_pt - pt
            norm_len = np.linalg.norm(tangent)
            if norm_len == 0: continue
            tangent /= norm_len
            normal = np.array([-tangent[1], tangent[0]])
            
            w_curr = widths[i]
            
            # Draw perpendicular line
            # Range: -3*sigma to +3*sigma
            span = int(3 * w_curr)
            
            for d in range(-span, span + 1):
                # Pixel coord
                px = int(pt[0] + d * normal[0])
                py = int(pt[1] + d * normal[1])
                
                if px < 0 or px >= self.w or py < 0 or py >= self.h:
                    continue
                
                # Distance from center line
                dist = abs(d) # Simple approximation, better would be sub-pixel
                
                # Gaussian Profile: I = I_max * exp(-x^2 / (2*sigma^2))
                val = np.exp(- (dist**2) / (2 * (w_curr**2)))
                
                # Accumulate (max)
                current_val = mask[py, px]
                new_val = val * intensity
                mask[py, px] = max(current_val, new_val)
                
        # 5. Apply high freq perlin noise texture to the stroke itself (optional "broken" effect)
        # Mask out random segments if very rough
        if roughness > 0.8:
            texture_noise = np.random.rand(self.h, self.w)
            # mask[texture_noise < 0.3] *= 0.2
            
        return mask

def test_generation():
    syn = ScratchSynthesizer(canvas_size=(512, 512))
    
    # Generate a few examples
    plt.figure(figsize=(15, 5))
    
    # 1. Smooth, thick scratch
    mask1 = syn.generate_scratch(width=5.0, roughness=0.1)
    plt.subplot(1, 3, 1)
    plt.imshow(mask1, cmap='gray')
    plt.title("Smooth, Thick (Width=5, Rough=0.1)")
    
    # 2. Examples of what user asked: Median roughness
    mask2 = syn.generate_scratch(width=2.5, roughness=0.5)
    plt.subplot(1, 3, 2)
    plt.imshow(mask2, cmap='gray')
    plt.title("Medium Roughness (Rough=0.5)")
    
    # 3. High roughness (broken edges)
    mask3 = syn.generate_scratch(width=2.0, roughness=1.0)
    plt.subplot(1, 3, 3)
    plt.imshow(mask3, cmap='gray')
    plt.title("High Roughness (Rough=1.0)")
    
    save_path = "/home/AI_CAC/results/plots/generated_scratches_demo.png"
    plt.savefig(save_path)
    print(f"Saved demo to {save_path}")

if __name__ == "__main__":
    test_generation()
