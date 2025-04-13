from PIL import Image
import cairosvg

# Path to the SVG file
svg_file = "logos/canada/canada_flag.svg"

# Output PNG file
png_file = "logos/canada/canada_flag.png"

# Convert SVG to PNG
cairosvg.svg2png(url=svg_file, write_to=png_file, output_width=300)

print(f"Converted {svg_file} to {png_file} with width 300px.")
