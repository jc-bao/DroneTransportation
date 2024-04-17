#!/bin/bash

# Script to extract 9 uniformly sampled frames from a GIF using FFmpeg and tile them into a 3x3 grid using ImageMagick.

# Check if an input file was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename.gif>"
    exit 1
fi

# Input GIF
input_gif="$1"

# Output file name for the tiled image
output_image="tiled_output.png"

# Temporary directory for intermediate files
temp_dir=$(mktemp -d -t gif-XXXXXX)

# Get total number of frames in the GIF using FFmpeg
total_frames=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 "$input_gif")

# Calculate step to sample 9 frames uniformly
step=$((total_frames / 9))

# Extract frames using FFmpeg
for (( i=0; i<9; i++ ))
do
    frame_number=$((i * step))
    ffmpeg -hide_banner -loglevel error -y -i "$input_gif" -vf "select='eq(n\,$frame_number)'" -vframes 1 "$temp_dir/frame_$i.png"
done

# Tile extracted frames into a 3x3 grid using ImageMagick
magick montage "$temp_dir/frame_*.png" -geometry +0+0 -tile 3x3 "$output_image"

# Clean up temporary files
rm -rf "$temp_dir"

echo "Tiled image created as $output_image"

