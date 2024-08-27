import glob
import os
import sys
import time

import cv2
import numpy as np
import png

from ip_basic import depth_map_utils
from ip_basic import vis_utils

import argparse
from tqdm import tqdm

def save_depth_image_as_uint16(depth_data_mm, output_depth_dir, depth_image_file_name, METER_SCALE=1000.0):
    """
    Converts a depth map in millimeters to a uint16 PNG image and saves it.
    The depth values in millimeters are first converted to meters, and then scaled by 256.0 before saving.

    Parameters:
    - depth_data_mm: np.array, the input depth map with float32 values in millimeters.
    - output_depth_dir: str, the directory where the output image should be saved.
    - depth_image_file_name: str, the name of the output PNG file.

    Returns:
    - None
    """
    # Convert millimeters to meters and then scale by 256
    depth_image = (depth_data_mm / METER_SCALE * 256.0).astype(np.uint16)
    
    # Construct the file path
    file_path = os.path.join(output_depth_dir, depth_image_file_name)
    
    # Save the uint16 image using pypng
    with open(file_path, 'wb') as f:
        writer = png.Writer(width=depth_image.shape[1],
                            height=depth_image.shape[0],
                            bitdepth=16,
                            greyscale=True)
        writer.write(f, depth_image)

def save_depth_as_npy_from_uint16(depth_data_mm, output_depth_dir, depth_image_file_name, METER_SCALE=1000.0):
    """
    Converts a depth map in uint16 PNG format to a numpy array in millimeters and saves it.
    The depth values are first scaled by 1000.0 and then converted to millimeters before saving.
    
    Parameters:
    - depth_data_mm: np.array, the input depth map with uint16 values.
    - output_depth_dir: str, the directory where the output numpy array should be saved.
    - depth_image_file_name: str, the name of the output numpy file.
    
    Returns:
    - None
    """
    # Convert uint16 to millimeters
    depth_data_mm = depth_data_mm.astype(np.float32) * METER_SCALE
    
    # Construct the file path
    file_path = os.path.join(output_depth_dir, depth_image_file_name)
    
    # Save the numpy array
    np.save(file_path, depth_data_mm)


def main(args):
    
    ## FILL TYPE: fast, multiscale
    
    # Fast fill with Gaussian blur @90Hz (paper result)
    # fill_type = 'fast'
    # extrapolate = False
    # blur_type = 'gaussian'

    # Fast Fill with bilateral blur, no extrapolation @87Hz (recommended)
    # fill_type = 'fast'
    # extrapolate = False
    # blur_type = 'bilateral'

    # Multi-scale dilations with extra noise removal, no extrapolation @ 30Hz
    # fill_type = 'multiscale'
    # extrapolate = False
    # blur_type = 'bilateral'

    fill_type = args.fill_type
    extrapolate = args.extrapolate
    blur_type = args.blur_type

    # Path to the dataset   
    input_depth_dir = args.data

    # Create temporary directories
    os.makedirs('tmp/tmp_png', exist_ok=True)
    os.makedirs('tmp/tmp_npy', exist_ok=True)

    # convert npy to png
    npy_files = os.listdir(input_depth_dir)
    npy_files = sorted([x for x in npy_files if (x.endswith('.npy'))])

    for npy_file in tqdm(npy_files):
        depth_data_mm = np.load(os.path.join(input_depth_dir, npy_file))
        save_depth_image_as_uint16(depth_data_mm, 'tmp/tmp_png', npy_file.replace('.npy', '.png'))

    save_output = True
    show_process = False
    save_depth_maps = True
    
    outputs_dir = args.output
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Get images in sorted order
    images_to_use = os.listdir('tmp/tmp_png')
    images_to_use = sorted([x for x in images_to_use if (x.endswith('.png'))])
    # np.random.seed(42)
    # np.random.shuffle(images_to_use)
    
    if args.limit is not None:
        images_to_use = images_to_use[:args.limit]
    
    # Rolling average array of times for time estimation
    avg_time_arr_length = 10
    last_fill_times = np.repeat([1.0], avg_time_arr_length)
    last_total_times = np.repeat([1.0], avg_time_arr_length)
    
    num_images = len(images_to_use)
    for i in range(num_images):

        depth_image_path = os.path.join(input_depth_dir,images_to_use[i])

        # Calculate average time with last n fill times
        avg_fill_time = np.mean(last_fill_times)
        avg_total_time = np.mean(last_total_times)

        # Show progress
        sys.stdout.write('\rProcessing {} / {}, '
                         'Avg Fill Time: {:.5f}s, '
                         'Avg Total Time: {:.5f}s, '
                         'Est Time Remaining: {:.3f}s'.format(
                             i, num_images - 1, avg_fill_time, avg_total_time,
                             avg_total_time * (num_images - i)))
        sys.stdout.flush()

        # Start timing
        start_total_time = time.time()

        # Load depth projections from uint16 image
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        projected_depths = np.float32(depth_image / 256.0)
              
        # Fill in
        start_fill_time = time.time()
        if fill_type == 'fast':
            final_depths = depth_map_utils.fill_in_fast(
                projected_depths, max_depth=args.max_depth, extrapolate=extrapolate, blur_type=blur_type)
        elif fill_type == 'multiscale':
            final_depths, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depths, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process, max_depth=args.max_depth)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))
        end_fill_time = time.time()

        # Display images from process_dict
        if fill_type == 'multiscale' and show_process:
            img_size = (570, 165)

            x_start = 80
            y_start = 50
            x_offset = img_size[0]
            y_offset = img_size[1]
            x_padding = 0
            y_padding = 28

            img_x = x_start
            img_y = y_start
            max_x = 1900

            row_idx = 0
            for key, value in process_dict.items():

                image_jet = cv2.applyColorMap(
                    np.uint8(value / np.amax(value) * 255),
                    cv2.COLORMAP_JET)
                vis_utils.cv2_show_image(
                    key, image_jet,
                    img_size, (img_x, img_y))

                img_x += x_offset + x_padding
                if (img_x + x_offset + x_padding) > max_x:
                    img_x = x_start
                    row_idx += 1
                img_y = y_start + row_idx * (y_offset + y_padding)

                # Save process images
                cv2.imwrite('process/' + key + '.png', image_jet)
                
            cv2.waitKey()

        # Save depth images to disk
        if save_depth_maps:
            depth_image_file_name = os.path.split(depth_image_path)[1]

            # Save depth map to a uint16 png (same format as disparity maps)
            file_path = outputs_dir + '/' + depth_image_file_name
            with open(file_path, 'wb') as f:
                depth_image = (final_depths * 256).astype(np.uint16)

                # pypng is used because cv2 cannot save uint16 format images
                writer = png.Writer(width=depth_image.shape[1],
                                    height=depth_image.shape[0],
                                    bitdepth=16,
                                    greyscale=True)
                writer.write(f, depth_image)

        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time
    

    png_files = os.listdir(outputs_dir)
    png_files = sorted([x for x in png_files if x.endswith('.png')])

    for png_file in tqdm(png_files):
        depth_image = cv2.imread(os.path.join(outputs_dir, png_file), cv2.IMREAD_ANYDEPTH)
        depth_image = np.float32(depth_image / 256.0)
        save_depth_as_npy_from_uint16(depth_image, 'tmp/tmp_npy', png_file.replace('.png', '.npy'))



if __name__ == "__main__":
    
    # Set up argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description='Run Depth Completion on a given dataset.')
    parser.add_argument('-data', type=str, default='/data/disk1/share/mnardon/nextmagDATA/DATASETS/lego/seq1_ALL', help='Path to the dataset')
    parser.add_argument('-limit', type=int, default=None, help='Limit the number of images to process')
    parser.add_argument('-output', type=str, default='tmp/output_png', help='Path to the output directory')
    parser.add_argument('-fill_type', type=str, default='multiscale', help='Type of fill to use (fast, multiscale)')
    parser.add_argument('-extrapolate', type=bool, default=False, help='Whether to extrapolate')
    parser.add_argument('-blur_type', type=str, default='bilateral', help='Type of blur to use (gaussian, bilateral)')
    parser.add_argument('-max_depth', type=float, default=100.0, help='Maximum depth to use for filling')

    # Parse the command-line arguments
    args = parser.parse_args()    
    
    main(args)
