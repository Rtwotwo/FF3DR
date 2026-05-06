"""
python /data2/dataset/Redal/work_feedforward_3drepo/running/utils/viz_utils.py \
    --input_dir /data2/dataset/Redal/work_feedforward_3drepo/dataset/MatrixCity/big_city_depth/aerial/train/big_high_block_1_depth \
    --output_path /data2/dataset/Redal/work_feedforward_3drepo/assets/big_high_block_1_depth.mp4 \
    --fps 25 \
    --crf 35 \
    --width 640
"""
import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Convert image sequence to video using FFmpeg')
    
    parser.add_argument('--input_dir', type=str, 
                        default='/data2/dataset/Redal/work_feedforward_3drepo/dataset/MOOM/01/imgs',
                        help='Path to the folder containing images')
    
    parser.add_argument('--output_path', type=str, 
                        default='output_video.mp4',
                        help='Path and filename for the output video')
    
    parser.add_argument('--fps', type=int, 
                        default=25, 
                        help='Frames per second of the output video')
    
    parser.add_argument('--crf', type=int, 
                        default=23, 
                        help='Video quality/size factor (0-51). Higher is smaller size.')
    
    parser.add_argument('--width', type=int, 
                        default=-1, 
                        help='Output video width. -1 keeps original size. e.g., 640 or 480.')

    args = parser.parse_args()

    # Get list of image files
    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    
    # Sort files
    sorted_files = sorted(files)
    
    print(f"Found {len(sorted_files)} images. Starting conversion with FFmpeg...")

    # Create a temporary file list for FFmpeg
    temp_list_path = "temp_file_list.txt"
    with open(temp_list_path, 'w') as f:
        for filename in sorted_files:
            safe_path = os.path.join(args.input_dir, filename).replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    # Build scale filter
    # If width is specified, use it and force height to be an even number (divisible by 2)
    # 'trunc(ow/a/2)*2' calculates height based on aspect ratio and rounds down to nearest even number
    scale_filter = '-1'
    if args.width > 0:
        scale_filter = f'{args.width}:trunc(ow/a/2)*2'

    # FFmpeg Command
    cmd = [
        'ffmpeg', '-y', 
        '-f', 'concat', '-safe', '0', '-i', temp_list_path,
        '-c:v', 'libx264', 
        '-crf', str(args.crf), 
        '-vf', f'scale={scale_filter}', 
        '-pix_fmt', 'yuv420p',
        '-r', str(args.fps),
        args.output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Video saved to: {args.output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred during video creation: {e}")
    finally:
        if os.path.exists(temp_list_path):
            os.remove(temp_list_path)

if __name__ == '__main__':
    main()