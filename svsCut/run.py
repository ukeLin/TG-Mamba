from __future__ import print_function
from optparse import OptionParser
import numpy as np

import os
import sys
import pydicom

from utils.DeepZoomStaticTiler import DeepZoomStaticTiler, xml_read_labels
from imageio import imsave
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

VIEWER_SLIDE_NAME = 'slide'

if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] <slide>')

    parser.add_option('-L', '--ignore-bounds', dest='limit_bounds',
                      default=True, action='store_false',
                      help='display entire scan area')
    parser.add_option('-e', '--overlap', metavar='PIXELS', dest='overlap',
                      type='int', default=0,
                      help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}', dest='format',
                      default='png',
                      help='image format for tiles [png]')
    parser.add_option('-j', '--jobs', metavar='COUNT', dest='workers',
                      type='int', default=8,
                      help='number of worker processes to start [4]')
    parser.add_option('-o', '--output', metavar='NAME', dest='basename',
                      help='base name of output file', default='./output')
    parser.add_option('-Q', '--quality', metavar='QUALITY', dest='quality',
                      type='int', default=90,
                      help='JPEG compression quality [90]')
    parser.add_option('-r', '--viewer', dest='with_viewer',
                      action='store_true',
                      help='generate directory tree with HTML viewer')
    # size h*w  224 or 512
    parser.add_option('-s', '--size', metavar='PIXELS', dest='tile_size',
                      type='int', default=512,
                      help='tile size [254, 512]')
    # 背景空白区域的面积
    parser.add_option('-B', '--Background', metavar='PIXELS', dest='Bkg',
                      type='float', default=100,
                      help='Max background threshold [50]; percentager of background allowed this paramter is small,data is small')

    parser.add_option('-x', '--xmlfile', metavar='NAME', dest='xmlfile',
                      help='xml file if needed')
    parser.add_option('-m', '--mask_type', metavar='COUNT', dest='mask_type',
                      type='int', default=1,
                      help='if xml file is used, keep tile within the ROI (1) or outside of it (0)')
    parser.add_option('-R', '--ROIpc', metavar='PIXELS', dest='ROIpc',
                      type='float', default=50,
                      help='To be used with xml file - minimum percentage of tile covered by ROI (white)')
    parser.add_option('-l', '--oLabelref', metavar='NAME', dest='oLabelref',
                      help='To be used with xml file - Only tile for label which contains the characters in oLabel')
    parser.add_option('-S', '--SaveMasks', metavar='NAME', dest='SaveMasks', default=False,
                      help='set to yes if you want to save ALL masks for ALL tiles (will be saved in same directory with <mask> suffix)')
    parser.add_option('-t', '--tmp_dcm', metavar='NAME', dest='tmp_dcm',
                      help='base name of output folder to save intermediate dcm images converted to jpg (we assume the patient ID is the folder name in which the dcm images are originally saved)')
    parser.add_option('-M', '--Mag', metavar='PIXELS', dest='Mag',
                      type='float', default=20,
                      help='Magnification at which tiling should be done (-1 of all)')
    parser.add_option('-N', '--normalize', metavar='NAME', dest='normalize',
                      help='if normalization is needed, N list the mean and std for each channel. For example \'57,22,-8,20,10,5\' with the first 3 numbers being the targeted means, and then the targeted stds')

    (opts, args) = parser.parse_args()

    try:
        # 指定文件夹路径
        slide_path = './data'

        # 使用 os.walk 递归查找所有 .svs 文件
        slides = []
        for root, dirs, files in os.walk(slide_path):
            for file in files:
                if file.endswith('.svs'):
                    slides.append(os.path.join(root, file))

        if not slides:
            print('No .svs files found in the specified directory or its subdirectories')

    except IndexError:
        parser.error('Missing slide argument')

    if opts.basename is None:
        # 如果opts.basename没有指定，使用文件夹名作为basename
        opts.basename = os.path.basename(slide_path)

    if opts.xmlfile is None:
        opts.xmlfile = ''

    try:
        if opts.normalize is not None:
            opts.normalize = [float(x) for x in opts.normalize.split(',')]
            if len(opts.normalize) != 6:
                opts.normalize = ''
                parser.error(
                    "ERROR: NO NORMALIZATION APPLIED: input vector does not have the right length - 6 values expected")
        else:
            opts.normalize = ''

    except:
        opts.normalize = ''
        parser.error("ERROR: NO NORMALIZATION APPLIED: input vector does not have the right format")

    ImgExtension = slide_path.split('*')[-1]

    slides = sorted(slides)
    for imgNb in range(len(slides)):
        filename = slides[imgNb]

        opts.basenameJPG = os.path.splitext(os.path.basename(filename))[0]
        print("processing: " + opts.basenameJPG)

        if "dcm" in ImgExtension:
            print("convert %s dcm to jpg" % filename)
            if opts.tmp_dcm is None:
                parser.error('Missing output folder for dcm>jpg intermediate files')
            elif not os.path.isdir(opts.tmp_dcm):
                parser.error('Missing output folder for dcm>jpg intermediate files')

            if filename[-3:] == 'jpg':
                continue

            ImageFile = pydicom.read_file(filename)
            im1 = ImageFile.pixel_array
            maxVal = float(im1.max())
            minVal = float(im1.min())
            height = im1.shape[0]
            width = im1.shape[1]
            image = np.zeros((height, width, 3), 'uint8')
            image[..., 0] = ((im1[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
            image[..., 1] = ((im1[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
            image[..., 2] = ((im1[:, :].astype(float) - minVal) / (maxVal - minVal) * 255.0).astype(int)
            dcm_ID = os.path.basename(os.path.dirname(filename))
            opts.basenameJPG = dcm_ID + "_" + opts.basenameJPG
            filename = os.path.join(opts.tmp_dcm, opts.basenameJPG + ".jpg")

            imsave(filename, image)

            output = os.path.join(opts.basename, opts.basenameJPG)

            try:
                DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds,
                                    opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG,
                                    opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks,
                                    opts.Mag, opts.normalize).run()
            except Exception as e:
                print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
                print(e)

        elif opts.xmlfile != '':
            xmldir = os.path.join(opts.xmlfile, opts.basenameJPG + '.xml')
            if os.path.isfile(xmldir):
                if (opts.mask_type == 1) or (opts.oLabelref != ''):
                    # either mask inside ROI, or mask outside but a reference label exist
                    xml_labels, xml_valid = xml_read_labels(xmldir)
                    if (opts.mask_type == 1):
                        # No inverse mask
                        Nbr_ROIs_ForNegLabel = 1
                    elif (opts.oLabelref != ''):
                        # Inverse mask and a label reference exist
                        Nbr_ROIs_ForNegLabel = 0

                    for oLabel in xml_labels:
                        if (opts.oLabelref in oLabel) or (opts.oLabelref == ''):
                            # is a label is identified
                            if (opts.mask_type == 0):
                                # Inverse mask and label exist in the image
                                Nbr_ROIs_ForNegLabel += 1
                                # there is a label, and map is to be inverted
                                output = os.path.join(opts.basename, oLabel + '_inv', opts.basenameJPG)
                                if not os.path.exists(os.path.join(opts.basename, oLabel + '_inv')):
                                    os.makedirs(os.path.join(opts.basename, oLabel + '_inv'))
                            else:
                                Nbr_ROIs_ForNegLabel += 1
                                output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
                                if not os.path.exists(os.path.join(opts.basename, oLabel)):
                                    os.makedirs(os.path.join(opts.basename, oLabel))
                            if 1:
                                try:
                                    DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap,
                                                        opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer,
                                                        opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type,
                                                        opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag,
                                                        opts.normalize).run()
                                except:
                                    print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

                        if Nbr_ROIs_ForNegLabel == 0:
                            print("label %s is not in that image; invert everything" % (opts.oLabelref))
                            # a label ref was given, and inverse mask is required but no ROI with this label in that map --> take everything
                            oLabel = opts.oLabelref
                            output = os.path.join(opts.basename, opts.oLabelref + '_inv', opts.basenameJPG)
                            if not os.path.exists(os.path.join(opts.basename, oLabel + '_inv')):
                                os.makedirs(os.path.join(opts.basename, oLabel + '_inv'))
                            if 1:
                                try:
                                    DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap,
                                                        opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer,
                                                        opts.Bkg, opts.basenameJPG, opts.xmlfile, opts.mask_type,
                                                        opts.ROIpc, oLabel, ImgExtension, opts.SaveMasks, opts.Mag,
                                                        opts.normalize).run()
                                except:
                                    print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))

                else:
                    # Background
                    oLabel = "non_selected_regions"
                    output = os.path.join(opts.basename, oLabel, opts.basenameJPG)
                    if not os.path.exists(os.path.join(opts.basename, oLabel)):
                        os.makedirs(os.path.join(opts.basename, oLabel))
                    try:
                        DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap,
                                            opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg,
                                            opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, oLabel,
                                            ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize).run()
                    except Exception as e:
                        print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
                        print(e)
            else:
                if (ImgExtension == ".jpg") | (ImgExtension == ".dcm"):
                    print("Input image to be tiled is jpg or dcm and not svs - will be treated as such")
                    output = os.path.join(opts.basename, opts.basenameJPG)
                    try:
                        DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap,
                                            opts.limit_bounds, opts.quality, opts.workers, opts.with_viewer, opts.Bkg,
                                            opts.basenameJPG, opts.xmlfile, opts.mask_type, opts.ROIpc, '',
                                            ImgExtension, opts.SaveMasks, opts.Mag, opts.normalize).run()
                    except Exception as e:
                        print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
                        print(e)
                else:
                    print("No xml file found for slide %s.svs (expected: %s). Directory or xml file does not exist" % (
                        opts.basenameJPG, xmldir))
                    continue
        else:
            # 没有XML文件的情况
            output = os.path.join(opts.basename, opts.basenameJPG)
            mag_str = f"{opts.Mag:.1f}"
            mag_folder = os.path.join(output, mag_str)

            # 检查子文件夹是否存在
            if os.path.exists(mag_folder):
                print(f"Image {opts.basenameJPG} already tiled at magnification {opts.Mag:.1f}")
                continue
            try:
                # if True:
                DeepZoomStaticTiler(filename, output, opts.format, opts.tile_size, opts.overlap, opts.limit_bounds,
                                    opts.quality, opts.workers, opts.with_viewer, opts.Bkg, opts.basenameJPG,
                                    opts.xmlfile, opts.mask_type, opts.ROIpc, '', ImgExtension, opts.SaveMasks,
                                    opts.Mag, opts.normalize).run()
            except Exception as e:
                print("Failed to process file %s, error: %s" % (filename, sys.exc_info()[0]))
                print(e)

    print("End")
