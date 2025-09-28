"""
Real-time GPU-accelerated image processing application using CUDA and PyCUDA.

This module provides a complete real-time image processing application that applies
various filters to camera input using GPU acceleration via CUDA kernels.
"""

import time

import cv2
import numpy as np
import pygame
import pycuda.autoinit  # pylint: disable=unused-import
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class RealTimeGPUFilterApp:
    """
    Real-time GPU-accelerated image processing application.

    This class provides a complete interface for applying various image filters
    to live camera input using CUDA kernelsif __name__ == "__main__":
    app = None
    try:
        app = RealTimeGPUFilterApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except (RuntimeError, ValueError, ImportError) as e:
        print(f"Application error: {e}")
    finally:
        if app is not None:
            app.cleanup()celeration.
    """

    def __init__(self):
        try:
            # Initialize camera with error handling
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                # Try alternative camera indices
                for i in range(1, 4):
                    self.cap = cv2.VideoCapture(i)
                    if self.cap.isOpened():
                        print(f"Using camera index {i}")
                        break
                else:
                    raise RuntimeError(
                        "No camera found. Please check your camera connection."
                    )

            self.width = 640
            self.height = 480
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Verify actual dimensions
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if actual_width != self.width or actual_height != self.height:
                print(
                    f"Warning: Camera using {actual_width}x{actual_height} "
                    f"instead of {self.width}x{self.height}"
                )
                self.width, self.height = actual_width, actual_height

            # Performance monitoring
            self.fps = 0
            self.frame_count = 0
            self.start_time = time.time()
            self.gpu_time = 0

            # CUDA timing events
            self.start_event = cuda.Event()
            self.end_event = cuda.Event()

            # Compile CUDA kernels
            self.compile_kernels()

            # Allocate GPU memory
            self.allocate_memory()

            # Initialize GUI
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Real-Time GPU Image Filters")

            # Available filters
            self.filters = [
                "Original",
                "Grayscale",
                "Sepia",
                "Edge Detection",
                "Gaussian Blur",
                "Emboss",
                "Negative",
                "Sketch",
                "Bilateral Filter",
                "Cartoon Effect",
                "Vignette Effect",
            ]
            self.current_filter = 0
            self._save_cooldown = 0

        except Exception as e:
            print(f"Initialization error: {e}")
            self.cleanup()
            raise

    def compile_kernels(self):
        """Compile all CUDA kernels for image processing filters."""
        self.mod = SourceModule(
            """
            __global__ void grayscale(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    unsigned char b = input[i];
                    unsigned char g = input[i + 1];
                    unsigned char r = input[i + 2];
                    
                    unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
                    output[i] = gray;
                    output[i + 1] = gray;
                    output[i + 2] = gray;
                }
            }
            
            __global__ void sepia(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    unsigned char b = input[i];
                    unsigned char g = input[i + 1];
                    unsigned char r = input[i + 2];
                    
                    int sepia_r = min(255, (int)(0.393f * r + 0.769f * g + 0.189f * b));
                    int sepia_g = min(255, (int)(0.349f * r + 0.686f * g + 0.168f * b));
                    int sepia_b = min(255, (int)(0.272f * r + 0.534f * g + 0.131f * b));
                    
                    output[i] = sepia_b;
                    output[i + 1] = sepia_g;
                    output[i + 2] = sepia_r;
                }
            }
            
            __global__ void edge_detection(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    // Handle borders by copying original values
                    if (idx < 1 || idx >= width - 1 || idy < 1 || idy >= height - 1) {
                        output[i] = input[i];
                        output[i + 1] = input[i + 1];
                        output[i + 2] = input[i + 2];
                        return;
                    }
                    
                    // Sobel X kernel
                    int gx = -input[((idy-1) * width + (idx-1)) * 3 + 1] + input[((idy-1) * width + (idx+1)) * 3 + 1] +
                             -2 * input[(idy * width + (idx-1)) * 3 + 1] + 2 * input[(idy * width + (idx+1)) * 3 + 1] +
                             -input[((idy+1) * width + (idx-1)) * 3 + 1] + input[((idy+1) * width + (idx+1)) * 3 + 1];
                    
                    // Sobel Y kernel
                    int gy = -input[((idy-1) * width + (idx-1)) * 3 + 1] - 2 * input[((idy-1) * width + idx) * 3 + 1] 
                             - input[((idy-1) * width + (idx+1)) * 3 + 1] + input[((idy+1) * width + (idx-1)) * 3 + 1] 
                             + 2 * input[((idy+1) * width + idx) * 3 + 1] + input[((idy+1) * width + (idx+1)) * 3 + 1];
                    
                    int magnitude = min(255, (int)sqrt((float)(gx * gx + gy * gy)));
                    output[i] = magnitude;
                    output[i + 1] = magnitude;
                    output[i + 2] = magnitude;
                }
            }
            
            __global__ void gaussian_blur(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    // Handle borders by copying original values
                    if (idx < 2 || idx >= width - 2 || idy < 2 || idy >= height - 2) {
                        output[i] = input[i];
                        output[i + 1] = input[i + 1];
                        output[i + 2] = input[i + 2];
                        return;
                    }
                    
                    // 5x5 Gaussian kernel
                    float kernel[25] = {1, 4, 7, 4, 1,
                                       4, 16, 26, 16, 4,
                                       7, 26, 41, 26, 7,
                                       4, 16, 26, 16, 4,
                                       1, 4, 7, 4, 1};
                    
                    float sum_b = 0, sum_g = 0, sum_r = 0;
                    for (int ky = -2; ky <= 2; ky++) {
                        for (int kx = -2; kx <= 2; kx++) {
                            int pixel_idx = ((idy + ky) * width + (idx + kx)) * 3;
                            float weight = kernel[(ky + 2) * 5 + (kx + 2)] / 273.0f;
                            sum_b += input[pixel_idx] * weight;
                            sum_g += input[pixel_idx + 1] * weight;
                            sum_r += input[pixel_idx + 2] * weight;
                        }
                    }
                    
                    output[i] = (unsigned char)sum_b;
                    output[i + 1] = (unsigned char)sum_g;
                    output[i + 2] = (unsigned char)sum_r;
                }
            }
            
            __global__ void emboss(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    // Handle borders by copying original values
                    if (idx < 1 || idx >= width - 1 || idy < 1 || idy >= height - 1) {
                        output[i] = input[i];
                        output[i + 1] = input[i + 1];
                        output[i + 2] = input[i + 2];
                        return;
                    }
                    
                    // Emboss kernel
                    int emboss_val = -input[((idy-1) * width + (idx-1)) * 3 + 1] 
                                   - input[((idy-1) * width + idx) * 3 + 1] 
                                   - input[(idy * width + (idx-1)) * 3 + 1] 
                                   + input[(idy * width + (idx+1)) * 3 + 1] 
                                   + input[((idy+1) * width + idx) * 3 + 1] 
                                   + input[((idy+1) * width + (idx+1)) * 3 + 1];
                    
                    emboss_val = max(0, min(255, emboss_val + 128));
                    output[i] = emboss_val;
                    output[i + 1] = emboss_val;
                    output[i + 2] = emboss_val;
                }
            }
            
            __global__ void negative(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    output[i] = 255 - input[i];
                    output[i + 1] = 255 - input[i + 1];
                    output[i + 2] = 255 - input[i + 2];
                }
            }
            
            __global__ void sketch(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    // Handle borders by copying original values
                    if (idx < 1 || idx >= width - 1 || idy < 1 || idy >= height - 1) {
                        output[i] = input[i];
                        output[i + 1] = input[i + 1];
                        output[i + 2] = input[i + 2];
                        return;
                    }
                    
                    // Convert to grayscale first
                    unsigned char gray = (unsigned char)(0.299f * input[i + 2] + 0.587f * input[i + 1] 
                                                         + 0.114f * input[i]);
                    
                    // Apply edge detection for sketch effect
                    int gx = -input[((idy-1) * width + (idx-1)) * 3 + 1] + input[((idy-1) * width + (idx+1)) * 3 + 1] 
                            + -2 * input[(idy * width + (idx-1)) * 3 + 1] + 2 * input[(idy * width + (idx+1)) * 3 + 1] 
                            + -input[((idy+1) * width + (idx-1)) * 3 + 1] + input[((idy+1) * width + (idx+1)) * 3 + 1];
                    
                    int gy = -input[((idy-1) * width + (idx-1)) * 3 + 1] - 2 * input[((idy-1) * width + idx) * 3 + 1] 
                           - input[((idy-1) * width + (idx+1)) * 3 + 1] + input[((idy+1) * width + (idx-1)) * 3 + 1] 
                           + 2 * input[((idy+1) * width + idx) * 3 + 1] + input[((idy+1) * width + (idx+1)) * 3 + 1];
                    
                    int magnitude = min(255, (int)sqrt((float)(gx * gx + gy * gy)));
                    unsigned char sketch_val = 255 - magnitude;
                    
                    output[i] = sketch_val;
                    output[i + 1] = sketch_val;
                    output[i + 2] = sketch_val;
                }
            }
            
            __global__ void vignette(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    float center_x = width / 2.0f;
                    float center_y = height / 2.0f;
                    float max_distance = sqrt(center_x * center_x + center_y * center_y);
                    float distance = sqrt((idx - center_x) * (idx - center_x) + (idy - center_y) * (idy - center_y));
                    
                    float vignette_factor = 1.0f - (distance / max_distance) * 0.7f;
                    vignette_factor = max(0.3f, vignette_factor);
                    
                    output[i] = (unsigned char)(input[i] * vignette_factor);
                    output[i + 1] = (unsigned char)(input[i + 1] * vignette_factor);
                    output[i + 2] = (unsigned char)(input[i + 2] * vignette_factor);
                }
            }
            
            __global__ void cartoon(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    // Quantize colors for cartoon effect
                    unsigned char b = input[i];
                    unsigned char g = input[i + 1];
                    unsigned char r = input[i + 2];
                    
                    // Reduce color depth
                    int levels = 8;
                    int scale = 256 / levels;
                    
                    r = ((r / scale) * scale);
                    g = ((g / scale) * scale);
                    b = ((b / scale) * scale);
                    
                    output[i] = b;
                    output[i + 1] = g;
                    output[i + 2] = r;
                }
            }
            
            __global__ void bilateral_filter(unsigned char *input, unsigned char *output, int width, int height) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                int idy = blockIdx.y * blockDim.y + threadIdx.y;
                
                if (idx < width && idy < height) {
                    int i = (idy * width + idx) * 3;
                    
                    // Handle borders by copying original values
                    if (idx < 2 || idx >= width - 2 || idy < 2 || idy >= height - 2) {
                        output[i] = input[i];
                        output[i + 1] = input[i + 1];
                        output[i + 2] = input[i + 2];
                        return;
                    }
                    
                    float sigma_d = 3.0f;  // Domain sigma
                    float sigma_r = 50.0f; // Range sigma
                    
                    float sum_b = 0, sum_g = 0, sum_r = 0;
                    float weight_sum = 0;
                    
                    unsigned char center_b = input[i];
                    unsigned char center_g = input[i + 1];
                    unsigned char center_r = input[i + 2];
                    
                    for (int dy = -2; dy <= 2; dy++) {
                        for (int dx = -2; dx <= 2; dx++) {
                            int neighbor_idx = ((idy + dy) * width + (idx + dx)) * 3;
                            
                            // Spatial weight
                            float spatial_weight = exp(-(dx * dx + dy * dy) / (2 * sigma_d * sigma_d));
                            
                            // Intensity weight
                            float diff_b = input[neighbor_idx] - center_b;
                            float diff_g = input[neighbor_idx + 1] - center_g;
                            float diff_r = input[neighbor_idx + 2] - center_r;
                            float intensity_diff = sqrt(diff_b * diff_b + diff_g * diff_g + diff_r * diff_r);
                            float intensity_weight = exp(-(intensity_diff * intensity_diff) / (2 * sigma_r * sigma_r));
                            
                            float weight = spatial_weight * intensity_weight;
                            
                            sum_b += input[neighbor_idx] * weight;
                            sum_g += input[neighbor_idx + 1] * weight;
                            sum_r += input[neighbor_idx + 2] * weight;
                            weight_sum += weight;
                        }
                    }
                    
                    output[i] = (unsigned char)(sum_b / weight_sum);
                    output[i + 1] = (unsigned char)(sum_g / weight_sum);
                    output[i + 2] = (unsigned char)(sum_r / weight_sum);
                }
            }
        """
        )

        # Get kernel functions
        self.grayscale_func = self.mod.get_function("grayscale")
        self.sepia_func = self.mod.get_function("sepia")
        self.edge_func = self.mod.get_function("edge_detection")
        self.blur_func = self.mod.get_function("gaussian_blur")
        self.emboss_func = self.mod.get_function("emboss")
        self.negative_func = self.mod.get_function("negative")
        self.sketch_func = self.mod.get_function("sketch")
        self.vignette_func = self.mod.get_function("vignette")
        self.cartoon_func = self.mod.get_function("cartoon")
        self.bilateral_func = self.mod.get_function("bilateral_filter")

    def allocate_memory(self):
        """Allocate GPU and pinned host memory for image processing."""
        # Allocate device memory for image processing
        self.image_size = self.width * self.height * 3  # RGB channels
        self.d_input = cuda.mem_alloc(self.image_size)
        self.d_output = cuda.mem_alloc(self.image_size)

        # Allocate pinned memory for faster transfers
        self.h_input = cuda.pagelocked_empty(
            (self.height, self.width, 3), dtype=np.uint8
        )
        self.h_output = cuda.pagelocked_empty(
            (self.height, self.width, 3), dtype=np.uint8
        )

        # Thread block and grid dimensions
        self.block_size = (16, 16, 1)
        self.grid_size = (
            (self.width + self.block_size[0] - 1) // self.block_size[0],
            (self.height + self.block_size[1] - 1) // self.block_size[1],
            1,
        )

    def run(self):
        """Run the main application loop with real-time image processing."""
        running = True
        clock = pygame.time.Clock()

        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RIGHT:
                        self.current_filter = (self.current_filter + 1) % len(
                            self.filters
                        )
                    elif event.key == pygame.K_LEFT:
                        self.current_filter = (self.current_filter - 1) % len(
                            self.filters
                        )
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Apply selected filter using CUDA
            filtered_frame = self.apply_filter(frame, self.current_filter)

            # Handle save after filtered_frame is created
            keys = pygame.key.get_pressed()
            if (
                keys[pygame.K_s]
                and hasattr(self, "_save_cooldown")
                and time.time() > self._save_cooldown
            ):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"filtered_{self.filters[self.current_filter].replace(' ', '_')}_{timestamp}.jpg"
                cv2.imwrite(filename, filtered_frame)
                print(f"Saved: {filename}")
                self._save_cooldown = time.time() + 1.0  # 1 second cooldown
            elif not hasattr(self, "_save_cooldown"):
                self._save_cooldown = 0

            # Display results
            self.display_frame(filtered_frame)

            # Show FPS and current filter
            self.display_info()

            # Limit FPS
            clock.tick(30)

        self.cleanup()

    def apply_filter(self, frame, filter_index):
        """Apply the selected CUDA filter to the input frame and return processed result."""
        # CUDA-accelerated filter application
        if filter_index == 0:  # Original
            return frame

        # Start GPU timing
        self.start_event.record()

        # Convert frame to contiguous array and use pinned memory
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.h_input[:] = frame_rgb
        frame_flat = self.h_input.flatten()

        # Copy to GPU (async copy with pinned memory)
        cuda.memcpy_htod(self.d_input, frame_flat)

        # Apply appropriate filter
        if filter_index == 1:  # Grayscale
            self.grayscale_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 2:  # Sepia
            self.sepia_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 3:  # Edge Detection
            self.edge_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 4:  # Gaussian Blur
            self.blur_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 5:  # Emboss
            self.emboss_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 6:  # Negative
            self.negative_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 7:  # Sketch
            self.sketch_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 8:  # Bilateral Filter
            self.bilateral_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 9:  # Cartoon Effect
            self.cartoon_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )
        elif filter_index == 10:  # Vignette Effect
            self.vignette_func(
                self.d_input,
                self.d_output,
                np.int32(self.width),
                np.int32(self.height),
                block=self.block_size,
                grid=self.grid_size,
            )

        # Copy result back to host using pinned memory
        cuda.memcpy_dtoh(self.h_output.flatten(), self.d_output)

        # End GPU timing
        self.end_event.record()
        self.end_event.synchronize()
        self.gpu_time = self.start_event.time_till(self.end_event)

        # Reshape and convert back to BGR
        result_rgb = self.h_output
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

        return result_bgr

    def display_frame(self, frame):
        """Convert frame to PyGame surface and display it on screen."""
        # Convert to PyGame surface and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        self.screen.blit(frame_surface, (0, 0))
        pygame.display.flip()

    def display_info(self):
        """Display FPS, current filter name, and user instructions on screen."""
        # Show FPS, filter name, and instructions
        font = pygame.font.Font(None, 36)

        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

        # Display current filter name
        filter_text = font.render(
            f"Filter: {self.filters[self.current_filter]}", True, (255, 255, 255)
        )
        self.screen.blit(filter_text, (10, 10))

        # Display FPS and GPU time
        fps_text = font.render(
            f"FPS: {self.fps:.1f} | GPU: {self.gpu_time:.2f}ms", True, (255, 255, 255)
        )
        self.screen.blit(fps_text, (10, 50))

        # Display instructions
        instruction_font = pygame.font.Font(None, 24)
        instructions = [
            "Left/Right Arrow: Change Filter",
            "S: Save Current Frame",
            "ESC: Exit",
        ]

        for i, instruction in enumerate(instructions):
            instruction_text = instruction_font.render(
                instruction, True, (255, 255, 255)
            )
            self.screen.blit(instruction_text, (10, self.height - 75 + i * 25))

    def cleanup(self):
        """Enhanced cleanup with error handling"""
        try:
            # Free GPU memory
            if hasattr(self, "d_input") and self.d_input:
                self.d_input.free()
                self.d_input = None
            if hasattr(self, "d_output") and self.d_output:
                self.d_output.free()
                self.d_output = None

            # Release camera
            if hasattr(self, "cap") and self.cap:
                self.cap.release()

            # Quit pygame
            pygame.quit()

            # Clear CUDA context
            cuda.Context.pop()

        except (AttributeError, RuntimeError) as e:
            print(f"Cleanup warning: {e}")

    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()
