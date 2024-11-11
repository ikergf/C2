import argparse
from skimage.io import imread, imsave

from inpainter import PSOInpainter_2


def main():
    args = parse_args()

    image = imread(args.input_image)
    mask = imread(args.mask, as_gray=True)

    # output_image = Inpainter(
    #     image,
    #     mask,
    #     patch_size=args.patch_size,
    #     plot_progress=args.plot_progress
    # ).inpaint()
    inpainter = PSOInpainter_2(
        image, 
        mask,
        patch_size = args.patch_size,
        plot_progress = args.plot_progress,
        num_particles = args.num_particles,
        w = args.weight,
        c1 = args.c1_parameter,
        c2 = args.c2_parameter)
    output_image = inpainter.inpaint()
    imsave(args.output, output_image, quality=100)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-ps',
        '--patch-size',
        help='the size of the patches',
        type=int,
        default=9
    )
    parser.add_argument(
        '-np',
        '--num_particles',
        help='number of particles',
        type=int,
        default=50
    )
    parser.add_argument(
        '-w',
        '--weight',
        help='weight',
        type=float,
        default=0.5
    )
    parser.add_argument(
        '-c1',
        '--c1_parameter',
        help='c1 parameter',
        type=float,
        default=2.0
    )
    parser.add_argument(
        '-c2',
        '--c2_parameter',
        help='c2 parameter',
        type=float,
        default=2.0
    )
    parser.add_argument(
        '-o',
        '--output',
        help='the file path to save the output image',
        default='output.jpg'
    )
    parser.add_argument(
        '--plot-progress',
        help='plot each generated image',
        action='store_true',
        default=False
    )
    parser.add_argument(
        'input_image',
        help='the image containing objects to be removed'
    )
    parser.add_argument(
        'mask',
        help='the mask of the region to be removed'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
