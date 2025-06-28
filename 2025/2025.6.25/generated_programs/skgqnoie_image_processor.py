
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import os

class ImageProcessor:
    def __init__(self, image_path):
        self.original_image = Image.open(image_path)
    
    def apply_filters(self, output_dir='image_outputs'):
        """Apply multiple image transformations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Original Image
        self.original_image.save(os.path.join(output_dir, 'original.png'))
        
        # Grayscale
        grayscale = self.original_image.convert('L')
        grayscale.save(os.path.join(output_dir, 'grayscale.png'))
        
        # Blur
        blurred = self.original_image.filter(ImageFilter.GaussianBlur(radius=5))
        blurred.save(os.path.join(output_dir, 'blurred.png'))
        
        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(self.original_image)
        high_contrast = enhancer.enhance(2.0)
        high_contrast.save(os.path.join(output_dir, 'high_contrast.png'))
        
        # Edge Detection
        edges = self.original_image.filter(ImageFilter.FIND_EDGES)
        edges.save(os.path.join(output_dir, 'edges.png'))
        
        # Color Histogram
        plt.figure(figsize=(15, 5))
        colors = ('r', 'g', 'b')
        channel_ids = (0, 1, 2)
        
        for channel_id, c in zip(channel_ids, colors):
            histogram = self.original_image.split()[channel_id].histogram()
            plt.subplot(1, 3, channel_id + 1)
            plt.title(f'{c.upper()} Channel Histogram')
            plt.bar(range(256), histogram, color=c, alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'color_histogram.png'))
        plt.close()
    
    def image_metadata(self):
        """Extract and return image metadata."""
        return {
            'format': self.original_image.format,
            'mode': self.original_image.mode,
            'size': self.original_image.size,
            'color_palette': str(self.original_image.getpalette()[:15]) if self.original_image.palette else 'No palette'
        }

def main():
    # For demonstration, create a sample image if no image exists
    if not os.path.exists('sample_image.png'):
        test_image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
        plt.imsave('sample_image.png', test_image)
    
    processor = ImageProcessor('sample_image.png')
    
    # Apply image filters
    processor.apply_filters()
    
    # Print metadata
    print(json.dumps(processor.image_metadata(), indent=2))

if __name__ == '__main__':
    main()
