"""
Visual Language Compressor V3 (High-Efficiency Edition)
1. Vector Quantization (K-Means) on image patches
2. Semantic Codebook Sorting (Luminance-based)
3. Delta (DPCM) Encoding for spatial coherence
4. LZMA compression (State-of-the-art entropy coding)
5. YCbCr 4:2:0 processing
"""

import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import lzma
import struct
import os
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings


@dataclass
class CompressionProfile:
    """Preset compression profiles balancing size vs quality."""

    name: str
    patch_size: int
    vocab_size: int
    chroma_subsample: bool
    use_dithering: bool


PROFILES = {
    "quality": CompressionProfile("quality", 2, 512, False, True),
    "balanced": CompressionProfile("balanced", 4, 256, True, True),
    "compact": CompressionProfile("compact", 8, 128, True, False),
    "extreme": CompressionProfile("extreme", 8, 64, True, False),
}


class VisualLanguageCompressorV3:
    # File format magic number
    MAGIC = b"VLC3"

    def __init__(self, profile: str = "balanced"):
        if profile not in PROFILES:
            raise ValueError(f"Unknown profile: {profile}")
        self.profile = PROFILES[profile]

    def _rgb_to_ycbcr(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to YCbCr using ITU-R BT.601."""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return np.stack([y, cb, cr], axis=-1).clip(0, 255).astype(np.uint8)

    def _ycbcr_to_rgb(self, ycbcr: np.ndarray) -> np.ndarray:
        """Convert YCbCr back to RGB."""
        # Explicitly cast to float32 to avoid uint8 wrapping/overflow during math
        y = ycbcr[:, :, 0].astype(np.float32)
        cb = ycbcr[:, :, 1].astype(np.float32)
        cr = ycbcr[:, :, 2].astype(np.float32)

        r = y + 1.402 * (cr - 128)
        g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
        b = y + 1.772 * (cb - 128)

        return np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)

    def _extract_patches(
        self, channel: np.ndarray, patch_size: int
    ) -> Tuple[np.ndarray, int, int]:
        h, w = channel.shape
        # Trim to multiple of patch size
        h_trim, w_trim = (h // patch_size) * patch_size, (w // patch_size) * patch_size
        channel = channel[:h_trim, :w_trim]

        # Extract patches using efficient reshaping
        patches = channel.reshape(
            h_trim // patch_size, patch_size, w_trim // patch_size, patch_size
        )
        patches = patches.swapaxes(1, 2).reshape(-1, patch_size * patch_size)
        return patches, h_trim, w_trim

    def _reconstruct_from_patches(
        self, patches: np.ndarray, h: int, w: int, patch_size: int
    ) -> np.ndarray:
        rows, cols = h // patch_size, w // patch_size
        patches = patches.reshape(rows, cols, patch_size, patch_size)
        return patches.swapaxes(1, 2).reshape(h, w)

    def _quantize_channel(
        self, patches: np.ndarray, n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Learn codebook and quantize patches."""
        # Use MiniBatchKMeans for speed
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=4096,
            n_init="auto",
            random_state=42,
            max_iter=100,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans.fit(patches)

        codebook = kmeans.cluster_centers_.astype(np.uint8)
        labels = kmeans.predict(patches)

        # --- IMPROVEMENT 1: Sort Codebook by Luminance ---
        # Sorting the codebook ensures that similar patches have similar indices.
        # This makes the token stream smooth (10, 11, 11, 12) rather than noisy (10, 255, 3, 100).
        # Smooth streams compress MUCH better with Delta encoding.
        patch_means = codebook.mean(axis=1)
        sorted_indices = np.argsort(patch_means)

        # Reorder codebook
        sorted_codebook = codebook[sorted_indices]

        # Create a mapping to remap the labels
        # inverse_map[old_id] = new_id
        inverse_map = np.zeros(n_clusters, dtype=int)
        inverse_map[sorted_indices] = np.arange(n_clusters)

        # Remap labels
        sorted_labels = inverse_map[labels]

        return sorted_codebook, sorted_labels

    def _compress_data_lzma(self, data: np.ndarray, is_tokens: bool = False) -> bytes:
        """
        Compress data using Delta Encoding + LZMA.
        """
        if is_tokens:
            # --- IMPROVEMENT 2: Delta Encoding (DPCM) ---
            # Instead of [10, 11, 12], store [10, +1, +1]
            # Since we sorted the codebook, neighbors are numerically close.
            # This creates a stream dominated by 0s and 1s.
            data = data.astype(np.int32)
            delta = np.diff(data, prepend=0)

            # Pack as flexible bytes based on range
            # Usually deltas are very small, fitting in int8
            if np.max(np.abs(delta)) < 128:
                payload = delta.astype(np.int8).tobytes()
            elif np.max(np.abs(delta)) < 32768:
                payload = delta.astype(np.int16).tobytes()
            else:
                payload = delta.astype(np.int32).tobytes()
        else:
            # Codebook compression
            payload = data.tobytes()

        # --- IMPROVEMENT 3: LZMA Compression ---
        # LZMA (Preset 9) is much stronger than zlib
        return lzma.compress(payload, preset=9)

    def _decompress_data_lzma(
        self, compressed: bytes, dtype, count: int, is_tokens: bool = False
    ) -> np.ndarray:
        """Decompress LZMA + Delta Decoding."""
        raw_bytes = lzma.decompress(compressed)

        if is_tokens:
            # Determine integer width from byte size
            total_bytes = len(raw_bytes)
            if total_bytes == count:
                deltas = np.frombuffer(raw_bytes, dtype=np.int8)
            elif total_bytes == count * 2:
                deltas = np.frombuffer(raw_bytes, dtype=np.int16)
            else:
                deltas = np.frombuffer(raw_bytes, dtype=np.int32)

            # Reverse Delta Encoding (Cumulative Sum)
            return np.cumsum(deltas).astype(dtype)
        else:
            return np.frombuffer(raw_bytes, dtype=dtype)

    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
        if mse == 0:
            return float("inf")
        return 10 * np.log10(255.0**2 / mse)

    def compress(self, image_path: str, output_path: str, verbose: bool = True) -> dict:
        if verbose:
            print(f"Compressing {image_path} [{self.profile.name}]...")

        # Load & Preprocess
        img = Image.open(image_path).convert("RGB")
        rgb = np.array(img)
        orig_h, orig_w = rgb.shape[:2]

        ps, vs = self.profile.patch_size, self.profile.vocab_size

        # --- FIX: Calculate Alignment Requirements ---
        if self.profile.chroma_subsample:
            chroma_ps = ps * 2
            chroma_vs = max(32, vs // 4)
        else:
            chroma_ps = ps
            chroma_vs = vs
        
        # We must trim the image to the largest block size used to keep channels aligned
        align_size = chroma_ps 
        h_trim = (orig_h // align_size) * align_size
        w_trim = (orig_w // align_size) * align_size
        
        # Trim RGB upfront so all channels cover the EXACT same pixels
        rgb = rgb[:h_trim, :w_trim]
        
        # Proceed with conversion on the aligned image
        ycbcr = self._rgb_to_ycbcr(rgb)

        # --- Process Y Channel ---
        # Note: We pass h_trim/w_trim to patches, but since we already trimmed 
        # the RGB source, _extract_patches won't cut anything further.
        y_patches, h, w = self._extract_patches(ycbcr[:, :, 0], ps)
        y_codebook, y_tokens = self._quantize_channel(y_patches, vs)

        # --- Process Chroma Channels ---
        cb_patches, cb_h, cb_w = self._extract_patches(ycbcr[:, :, 1], chroma_ps)
        cr_patches, _, _ = self._extract_patches(ycbcr[:, :, 2], chroma_ps)

        cb_codebook, cb_tokens = self._quantize_channel(cb_patches, chroma_vs)
        cr_codebook, cr_tokens = self._quantize_channel(cr_patches, chroma_vs)

        # --- Compression Step ---
        # Compress tokens (using delta encoding strategy)
        y_tok_comp = self._compress_data_lzma(y_tokens, is_tokens=True)
        cb_tok_comp = self._compress_data_lzma(cb_tokens, is_tokens=True)
        cr_tok_comp = self._compress_data_lzma(cr_tokens, is_tokens=True)

        # Compress codebooks
        y_cb_comp = self._compress_data_lzma(y_codebook)
        cb_cb_comp = self._compress_data_lzma(cb_codebook)
        cr_cb_comp = self._compress_data_lzma(cr_codebook)

        # Write to file
        with open(output_path, "wb") as f:
            f.write(self.MAGIC)
            # Header
            header = struct.pack(
                "<HHHHHHHHHHB",
                orig_h,
                orig_w,
                h,
                w,
                cb_h,
                cb_w,
                ps,
                chroma_ps,
                vs,
                chroma_vs,
                1 if self.profile.chroma_subsample else 0,
            )
            f.write(header)

            # Lengths
            f.write(
                struct.pack(
                    "<IIIIII",
                    len(y_cb_comp),
                    len(cb_cb_comp),
                    len(cr_cb_comp),
                    len(y_tok_comp),
                    len(cb_tok_comp),
                    len(cr_tok_comp),
                )
            )

            # Payload
            f.write(y_cb_comp + cb_cb_comp + cr_cb_comp)
            f.write(y_tok_comp + cb_tok_comp + cr_tok_comp)

        # Stats
        orig_size = orig_h * orig_w * 3
        comp_size = os.path.getsize(output_path)

        if verbose:
            print(f"  Ratio: {orig_size/comp_size:.2f}x ({comp_size/1024:.1f} KB)")

        return {"ratio": orig_size / comp_size, "compressed_size": comp_size}

    def decompress(self, compressed_path: str, output_path: str, verbose: bool = True):
        if verbose:
            print(f"Decompressing {compressed_path}...")

        with open(compressed_path, "rb") as f:
            if f.read(4) != self.MAGIC:
                raise ValueError("Invalid Magic")

            # Header
            (
                orig_h,
                orig_w,
                y_h,
                y_w,
                cb_h,
                cb_w,
                y_ps,
                cb_ps,
                y_vs,
                cb_vs,
                is_subsampled,
            ) = struct.unpack("<HHHHHHHHHHB", f.read(21))

            # Lengths
            lens = struct.unpack("<IIIIII", f.read(24))

            # Read Blobs
            y_cb_b = f.read(lens[0])
            cb_cb_b = f.read(lens[1])
            cr_cb_b = f.read(lens[2])
            y_tok_b = f.read(lens[3])
            cb_tok_b = f.read(lens[4])
            cr_tok_b = f.read(lens[5])

        # Decompress Codebooks
        y_codebook = self._decompress_data_lzma(y_cb_b, np.uint8, -1).reshape(
            y_vs, y_ps**2
        )
        cb_codebook = self._decompress_data_lzma(cb_cb_b, np.uint8, -1).reshape(
            cb_vs, cb_ps**2
        )
        cr_codebook = self._decompress_data_lzma(cr_cb_b, np.uint8, -1).reshape(
            cb_vs, cb_ps**2
        )

        # Decompress Tokens
        num_y = (y_h // y_ps) * (y_w // y_ps)
        num_cb = (cb_h // cb_ps) * (cb_w // cb_ps)

        y_tokens = self._decompress_data_lzma(y_tok_b, np.int32, num_y, is_tokens=True)
        cb_tokens = self._decompress_data_lzma(
            cb_tok_b, np.int32, num_cb, is_tokens=True
        )
        cr_tokens = self._decompress_data_lzma(
            cr_tok_b, np.int32, num_cb, is_tokens=True
        )

        # Reconstruct
        y = self._reconstruct_from_patches(y_codebook[y_tokens], y_h, y_w, y_ps)
        cb = self._reconstruct_from_patches(cb_codebook[cb_tokens], cb_h, cb_w, cb_ps)
        cr = self._reconstruct_from_patches(cr_codebook[cr_tokens], cb_h, cb_w, cb_ps)

        # Upsample Chroma if needed
        if is_subsampled:
            cb = np.array(Image.fromarray(cb).resize((y_w, y_h), Image.BILINEAR))
            cr = np.array(Image.fromarray(cr).resize((y_w, y_h), Image.BILINEAR))

        rgb = self._ycbcr_to_rgb(np.stack([y, cb, cr], axis=-1))

        if rgb.shape[:2] != (orig_h, orig_w):
            rgb = np.array(Image.fromarray(rgb).resize((orig_w, orig_h), Image.LANCZOS))

        Image.fromarray(rgb).save(output_path)
        if verbose:
            print(f"  Saved to {output_path}")


# --- Comparison Script ---
def run_comparison(image_path):
    import time

    print(f"Processing: {image_path}")

    # Get base filename (e.g. "photo.jpg" -> "photo")
    base_name = os.path.splitext(image_path)[0]

    results = []
    for pname in PROFILES:
        # V3 Compression
        comp = VisualLanguageCompressorV3(pname)
        start = time.time()

        # Compress to a temporary binary file
        temp_bin = "temp.vlc3"
        c_stats = comp.compress(image_path, temp_bin, verbose=False)
        comp_time = time.time() - start

        # Decompress to a specific filename for inspection
        output_png = f"{base_name}_{pname}.png"
        comp.decompress(temp_bin, output_png, verbose=False)

        # Metrics
        orig = np.array(Image.open(image_path).convert("RGB"))
        rec = np.array(Image.open(output_png).convert("RGB"))
        psnr = comp.calculate_psnr(orig, rec)

        results.append(
            (pname, c_stats["ratio"], c_stats["compressed_size"], psnr, comp_time)
        )
        print(f"  [{pname}] Saved to: {output_png}")

    print(
        f"\n{'Profile':<10} {'Ratio':>8} {'Size (KB)':>10} {'PSNR (dB)':>10} {'Time (s)':>8}"
    )
    print("-" * 55)
    for r in results:
        print(f"{r[0]:<10} {r[1]:>7.1f}x {r[2]/1024:>9.1f} {r[3]:>10.2f} {r[4]:>8.2f}")

    # Cleanup only the temp binary file
    if os.path.exists("temp.vlc3"):
        os.remove("temp.vlc3")


if __name__ == "__main__":
    import sys

    # Generate test image if none provided
    target_img = "test_image.png"
    if len(sys.argv) > 1:
        target_img = sys.argv[1]

    # Create a dummy image if it doesn't exist so the script runs out of the box
    if not os.path.exists(target_img):
        print(f"Generating dummy image: {target_img}")
        arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        Image.fromarray(arr).save(target_img)

    run_comparison(target_img)
