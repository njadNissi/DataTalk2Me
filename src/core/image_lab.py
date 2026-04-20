import numpy
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import rembg
import io
from enum import Enum
from streamlit_cropper import st_cropper

# --------------------------
# ENUMS & HELPER FUNCTIONS
# --------------------------
class FilterEffect(Enum):
    ORIGINAL = "Original"
    VINTAGE = "Vintage"
    POPART = "Pop Art"
    BLACK_WHITE = "Black & White"
    SEPIA = "Sepia"
    COOL_TONE = "Cool Tone"
    WARM_TONE = "Warm Tone"
    HDR = "HDR Effect"
    SKETCH = "Pencil Sketch"
    CARTOON = "Cartoon"
    NEON = "Neon Glow"
    BLUR = "Gaussian Blur"
    SHARPEN = "Sharpen"
    EMBOSS = "Emboss"
    EDGE_DETECT = "Edge Detect"
    POSTERIZE = "Posterize"
    SOLARIZE = "Solarize"
    INVERT = "Invert Colors"
    PASTEL = "Pastel"
    WATERCOLOR = "Watercolor"


def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes for download/previews"""
    buf = io.BytesIO()
    img.save(buf, format=format, quality=95)
    buf.seek(0)
    return buf

def apply_filter(img: Image.Image, effect: FilterEffect) -> Image.Image:
    """Apply creative filters/effects to image"""
    if effect == FilterEffect.ORIGINAL:
        return img
    
    # Basic adjustments
    enhancer_bright = ImageEnhance.Brightness(img)
    enhancer_contrast = ImageEnhance.Contrast(img)
    enhancer_color = ImageEnhance.Color(img)
    enhancer_sharp = ImageEnhance.Sharpness(img)

    if effect == FilterEffect.VINTAGE:
        img = enhancer_color.enhance(0.7)
        img = enhancer_contrast.enhance(1.2)
        img = enhancer_bright.enhance(0.9)
        return img.filter(ImageFilter.GaussianBlur(1))
    
    elif effect == FilterEffect.POPART:
        img = ImageOps.posterize(img, 3)
        return enhancer_contrast.enhance(2.0)
    
    elif effect == FilterEffect.BLACK_WHITE:
        return ImageOps.grayscale(img)
    
    elif effect == FilterEffect.SEPIA:
        sepia = ImageOps.grayscale(img)
        sepia = ImageEnhance.Color(sepia).enhance(0.2)
        sepia = ImageEnhance.Brightness(sepia).enhance(1.1)
        return sepia
    
    elif effect == FilterEffect.COOL_TONE:
        img_np = np.array(img)
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] + 30, 0, 255)  # Boost blue
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] - 10, 0, 255)  # Reduce red
        return Image.fromarray(img_np)
    
    elif effect == FilterEffect.WARM_TONE:
        img_np = np.array(img)
        img_np[:, :, 0] = np.clip(img_np[:, :, 0] + 30, 0, 255)  # Boost red
        img_np[:, :, 2] = np.clip(img_np[:, :, 2] - 10, 0, 255)  # Reduce blue
        return Image.fromarray(img_np)
    
    elif effect == FilterEffect.HDR:
        return enhancer_contrast.enhance(1.8).filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    
    elif effect == FilterEffect.SKETCH:
        gray = ImageOps.grayscale(img)
        inverted = ImageOps.invert(gray)
        blurred = inverted.filter(ImageFilter.GaussianBlur(10))
        sketch = ImageOps.invert(blurred)
        return Image.blend(gray, sketch, 0.5)
    
    elif effect == FilterEffect.CARTOON:
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9
        )
        color = cv2.bilateralFilter(img_np, 9, 300, 300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return Image.fromarray(cartoon)
    
    elif effect == FilterEffect.NEON:
        img_np = np.array(img)
        img_np = cv2.GaussianBlur(img_np, (1,1), 0)
        img_np = cv2.Canny(img_np, 100, 200)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        img_np[:, :, 2] = 255  # Red channel to max for neon effect
        return Image.fromarray(img_np)
    
    elif effect == FilterEffect.BLUR:
        return img.filter(ImageFilter.GaussianBlur(5))
    
    elif effect == FilterEffect.SHARPEN:
        return img.filter(ImageFilter.UnsharpMask(radius=3, percent=200))
    
    elif effect == FilterEffect.EMBOSS:
        return img.filter(ImageFilter.EMBOSS)
    
    elif effect == FilterEffect.EDGE_DETECT:
        return img.filter(ImageFilter.FIND_EDGES)
    
    elif effect == FilterEffect.POSTERIZE:
        return ImageOps.posterize(img, 4)
    
    elif effect == FilterEffect.SOLARIZE:
        return ImageOps.solarize(img, threshold=128)
    
    elif effect == FilterEffect.INVERT:
        return ImageOps.invert(img)
    
    elif effect == FilterEffect.PASTEL:
        img = enhancer_bright.enhance(1.2)
        img = enhancer_contrast.enhance(0.8)
        return img.filter(ImageFilter.GaussianBlur(2))
    
    elif effect == FilterEffect.WATERCOLOR:
        img = enhancer_sharp.enhance(0.5)
        img = enhancer_contrast.enhance(1.1)
        return img.filter(ImageFilter.MedianFilter(size=3))
