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
    fill_type = 'multiscale'
    extrapolate = False
    blur_type = 'bilateral'





    
    
    input_depth_dir = args.data
    
    save_output = True
    show_process = True
    save_depth_maps = True
    
    this_file_path = os.path.dirname(os.path.realpath(__file__))
    outputs_dir = args.output
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Get images in sorted order
    images_to_use = os.listdir(input_depth_dir)
    # images_to_use = sorted([x for x in images_to_use if (x.endswith('.png') and x.startswith('depth'))])
    images_to_use = sorted([x for x in images_to_use if (x.endswith('.png'))])
    np.random.seed(43)
    np.random.shuffle(images_to_use)
    
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
        
        # # Load depth projections from numpy file
        # projected_depths = np.load(depth_image_path).astype(np.float32)
        # print(f'Loaded {depth_image_path} with shape {projected_depths.shape}') 
        # print(f'Min: {np.min(projected_depths)}, Max: {np.max(projected_depths)}')
        # print(f'Mean: {np.mean(projected_depths)}, Std: {np.std(projected_depths)}')
        # print(f'Median: {np.median(projected_depths)}')
        # print(f'{projected_depths[400:410,400:410]}')
        
        # Fill in
        start_fill_time = time.time()
        if fill_type == 'fast':
            final_depths = depth_map_utils.fill_in_fast(
                projected_depths, max_depth=100, extrapolate=extrapolate, blur_type=blur_type)
        elif fill_type == 'multiscale':
            final_depths, process_dict = depth_map_utils.fill_in_multiscale(
                projected_depths, extrapolate=extrapolate, blur_type=blur_type,
                show_process=show_process)
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
                # vis_utils.cv2_show_image(
                #     key, image_jet,
                #     img_size, (img_x, img_y))

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
                
            # Save final_depths with open cv in a greyscale image
            # cv2.imwrite(file_path.replace('.png', '_cv.png'), final_depths)
            # save file in numpy format
            np.save(file_path.replace('.png', '.npy'), final_depths)
            
            # print(final_depths.shape)
            # print(final_depths)
            # MIN = 40 
            # MAX = 85
            # mu = 0.1
            # depth_image = np.clip(depth_image, MIN * (1 - mu), MAX * (1 + mu))
            # normalized_depth = (((depth_image - (MIN * (1 - mu))) / ((MAX * (1 + mu)) - (MIN * (1 - mu)))) * 255.0)

            # # Ensure there are no NaN or infinite values and clip values to the range [0, 255]
            # normalized_depth = np.nan_to_num(normalized_depth, nan=0.0, posinf=255.0, neginf=0.0)
            # normalized_depth = np.clip(normalized_depth, 0, 255)

            # # Convert to uint8 and create an RGB image
            # grayscale_image = normalized_depth.astype(np.uint8)
            # rgb_image = np.squeeze(np.stack([grayscale_image] * 3, axis=-1))
            # cv2.imwrite(os.path.join(outputs_dir,f'tmp{i}.png'), rgb_image)
            

        end_total_time = time.time()

        # Update fill times
        last_fill_times = np.roll(last_fill_times, -1)
        last_fill_times[-1] = end_fill_time - start_fill_time

        # Update total times
        last_total_times = np.roll(last_total_times, -1)
        last_total_times[-1] = end_total_time - start_total_time
    
if __name__ == "__main__":
    
    # Set up argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description='Run Depth Completion on a given dataset.')
    parser.add_argument('-data', type=str, default='/data/disk1/share/mnardon/nextmagDATA/DATASETS/lego/seq1_ALL', help='Path to the dataset')
    parser.add_argument('-limit', type=int, default=None, help='Limit the number of images to process')
    parser.add_argument('-output', type=str, default='output_tmp', help='Path to the output directory')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    
    
    
    
    main(args)
