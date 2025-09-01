from __future__ import print_function

from unicodedata import normalize
import numpy as np
import scipy.misc
import os

# 设置openslide二进制文件的路径
os.add_dll_directory(r'D:\TMB\WSI切分\svsCut\openslide_dll\openslide-bin-4.0.0.3-windows-x64\bin')
import openslide
import sys
from imageio import imread
from xml.dom import minidom
from PIL import Image, ImageDraw


class DeepZoomImageTiler(object):
    """处理单个图像的瓦片生成和元数据"""

    def __init__(self, dz, basename, format, associated, queue, slide, basenameJPG, xmlfile, mask_type, xmlLabel, ROIpc,
                 ImgExtension, SaveMasks, Mag, normalize):
        """
            初始化DeepZoomImageTiler类。
            Args:
                dz (openslide.OpenSlide): OpenSlide对象，用于访问图像数据。
                basename (str): 输出文件的基本名称。
                format (str): 瓦片的图像格式（如'jpeg', 'png'）。
                associated (str): 关联信息或标识符（如果适用）。
                queue (queue.Queue): 用于异步处理瓦片的队列。
                slide (openslide.OpenSlide): OpenSlide对象，用于瓦片访问。
                basenameJPG (str): JPG版本图像的基本名称（如果适用）。
                xmlfile (str): XML文件的路径，包含注释信息（如果适用）。
                mask_type (int): 掩模类型，决定是否在ROI内部或外部保留瓦片。
                xmlLabel (str): 用于过滤XML中注释的标签名（如果适用）。
                ROIpc (float): 瓦片被ROI覆盖的最小百分比。
                ImgExtension (str): 输入图像文件的扩展名。
                SaveMasks (bool): 是否保存所有瓦片的掩模。
                Mag (float): 进行瓦片处理的放大倍数。
                normalize (str): 归一化参数（均值和标准差，用于每个通道）。
        """
        self._dz = dz
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._format = format
        self._associated = associated
        self._queue = queue
        self._processed = 0
        self._slide = slide
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._xmlLabel = xmlLabel
        self._ROIpc = ROIpc
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize

    def run(self):
        """运行瓦片生成和DZI元数据写入过程。"""
        self._write_tiles()
        self._write_dzi()

    def _write_tiles(self):
        """为图像写入瓦片"""
        # 使用命令行参数或默认设置指定的放大倍数
        Magnification = 20
        # tol = 2
        # Factors 是图像金字塔各层的降采样因子
        Factors = self._slide.level_downsamples
        try:
            # 从图像属性中获取目标放大倍数
            Objective = float(self._slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        except:
            print(self._basename + " - No Obj information found")
            print(self._ImgExtension)
            if ("jpg" in self._ImgExtension) | ("dcm" in self._ImgExtension) | ("tif" in self._ImgExtension):
                Objective = 1.
                Magnification = Objective
                print("input is jpg - will be tiled as such with %f" % Objective)
            else:
                return
        # 根据目标放大倍数和降采样因子计算可用的放大倍数
        Available = tuple(Objective / x for x in Factors)
        # 找到大于等于指定放大倍数的最高可用放大倍数
        Mismatch = tuple(x - Magnification for x in Available)
        AbsMismatch = tuple(abs(x) for x in Mismatch)
        if len(AbsMismatch) < 1:
            print(self._basename + " - Objective field empty!")
            return
        xml_valid = False

        if True:
            ImgID = os.path.basename(self._basename)
            xmldir = os.path.join(self._xmlfile, ImgID + '.xml')
            # 根据文件扩展名和XML文件的存在情况读取掩膜
            if (self._xmlfile != '') & (self._ImgExtension != 'jpg') & (self._ImgExtension != 'dcm'):
                mask, xml_valid, Img_Fact = self.xml_read(xmldir, self._xmlLabel)
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return
            elif (self._xmlfile != '') & (self._ImgExtension == 'dcm'):
                mask, xml_valid, Img_Fact = self.jpg_mask_read(xmldir)
                if xml_valid == False:
                    print("Error: xml %s file cannot be read properly - please check format" % xmldir)
                    return
            # 遍历金字塔的每一层
            for level in range(self._dz.level_count - 1, -1, -1):
                ThisMag = Available[0] / pow(2, self._dz.level_count - (level + 1))
                # 如果指定了放大倍数且当前层不匹配
                if self._Mag > 0:
                    if ThisMag != self._Mag:
                        continue

                tiledir = os.path.join("%s" % self._basename, str(ThisMag))
                if not os.path.exists(tiledir):
                    os.makedirs(tiledir)
                cols, rows = self._dz.level_tiles[level]
                # 遍历当前层的每一个瓦片
                for row in range(rows):
                    for col in range(cols):
                        InsertBaseName = False
                        if InsertBaseName:
                            tilename = os.path.join(tiledir, '%s_%d_%d.%s' % (
                                self._basenameJPG, col, row, self._format))
                            tilename_bw = os.path.join(tiledir, '%s_%d_%d_mask.%s' % (
                                self._basenameJPG, col, row, self._format))
                        else:
                            tilename = os.path.join(tiledir, '%d_%d.%s' % (
                                col, row, self._format))
                            tilename_bw = os.path.join(tiledir, '%d_%d_mask.%s' % (
                                col, row, self._format))
                        # 计算瓦片的掩膜和掩膜百分比
                        if xml_valid:
                            Dlocation, Dlevel, Dsize = self._dz.get_tile_coordinates(level, (col, row))
                            Ddimension = tuple([pow(2, (self._dz.level_count - 1 - level)) * x for x in
                                                self._dz.get_tile_dimensions(level, (col, row))])
                            startIndY_current_level_conv = (int((Dlocation[1]) / Img_Fact))
                            endIndY_current_level_conv = (int((Dlocation[1] + Ddimension[1]) / Img_Fact))
                            startIndX_current_level_conv = (int((Dlocation[0]) / Img_Fact))
                            endIndX_current_level_conv = (int((Dlocation[0] + Ddimension[0]) / Img_Fact))
                            TileMask = mask[startIndY_current_level_conv:endIndY_current_level_conv,
                                       startIndX_current_level_conv:endIndX_current_level_conv]
                            PercentMasked = mask[startIndY_current_level_conv:endIndY_current_level_conv,
                                            startIndX_current_level_conv:endIndX_current_level_conv].mean()

                            if self._mask_type == 0:
                                # keep ROI outside of the mask
                                PercentMasked = 1.0 - PercentMasked

                        else:
                            PercentMasked = 1.0
                            TileMask = []

                        if not os.path.exists(tilename):
                            self._queue.put((self._associated, level, (col, row),
                                             tilename, self._format, tilename_bw, PercentMasked, self._SaveMasks,
                                             TileMask, self._normalize))
                        self._tile_done()

    def _tile_done(self):
        """更新已处理的瓦片计数并打印进度信息。"""
        self._processed += 1
        count, total = self._processed, self._dz.tile_count
        if count % 100 == 0 or count == total:
            print("Tiling %s: wrote %d/%d tiles" % (
                self._associated or 'slide', count, total),
                  end='\r', file=sys.stderr)
            if count == total:
                print(file=sys.stderr)

    def _write_dzi(self):
        """写入Deep Zoom Image (DZI)元数据文件"""
        with open('%s.dzi' % self._basename, 'w') as fh:
            fh.write(self.get_dzi())

    def get_dzi(self):
        """返回DZI元数据作为字符串"""
        return self._dz.get_dzi(self._format)

    def jpg_mask_read(self, xmldir):
        """
            读取JPEG掩模文件并返回掩模数组。

            Args:
                xmldir (str): XML文件的路径，其扩展名将被更改为'mask.jpg'以找到掩模文件。

            Returns:
                tuple: 包含掩模数组、XML有效性标志和图像因子的元组。
        """
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]
        # Img_Fact = int(ImgMaxSizeX_orig / 1.0 / cols)
        Img_Fact = 1
        try:
            # xmldir: change extension from xml to *jpg
            xmldir = xmldir[:-4] + "mask.jpg"
            # xmlcontent = read xmldir image
            xmlcontent = imread(xmldir)
            xmlcontent = xmlcontent - np.min(xmlcontent)
            mask = xmlcontent / np.max(xmlcontent)
            # we want image between 0 and 1
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        return mask, xml_valid, Img_Fact

    def xml_read(self, xmldir, Attribute_Name):
        """
            读取XML文件并生成掩模。

            Args:
                xmldir (str): XML文件的路径。
                Attribute_Name (str): 用于过滤注释的属性名。

            Returns:
                tuple: 包含掩模数组、XML有效性标志和新图像因子的元组。
        """
        # Original size of the image
        ImgMaxSizeX_orig = float(self._dz.level_dimensions[-1][0])
        ImgMaxSizeY_orig = float(self._dz.level_dimensions[-1][1])
        # Number of centers at the highest resolution
        cols, rows = self._dz.level_tiles[-1]

        NewFact = max(ImgMaxSizeX_orig, ImgMaxSizeY_orig) / min(max(ImgMaxSizeX_orig, ImgMaxSizeY_orig), 15000.0)

        Img_Fact = float(ImgMaxSizeX_orig) / 5.0 / float(cols)

        try:
            xmlcontent = minidom.parse(xmldir)
            xml_valid = True
        except:
            xml_valid = False
            print("error with minidom.parse(xmldir)")
            return [], xml_valid, 1.0

        xy = {}
        xy_neg = {}
        NbRg = 0
        labelIDs = xmlcontent.getElementsByTagName('Annotation')
        for labelID in labelIDs:
            if (Attribute_Name == []) | (Attribute_Name == ''):
                isLabelOK = True
            else:
                try:
                    labeltag = labelID.getElementsByTagName('Attribute')[0]
                    if Attribute_Name == labeltag.attributes['Value'].value:
                        isLabelOK = True
                    else:
                        isLabelOK = False
                except:
                    isLabelOK = False
            if Attribute_Name == "non_selected_regions":
                isLabelOK = True

            if isLabelOK:
                regionlist = labelID.getElementsByTagName('Region')
                for region in regionlist:
                    vertices = region.getElementsByTagName('Vertex')
                    NbRg += 1
                    regionID = region.attributes['Id'].value + str(NbRg)
                    NegativeROA = region.attributes['NegativeROA'].value
                    if len(vertices) > 0:
                        if NegativeROA == "0":
                            xy[regionID] = []
                            for vertex in vertices:
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy[regionID].append((x, y))

                        elif NegativeROA == "1":
                            xy_neg[regionID] = []
                            for vertex in vertices:
                                x = int(round(float(vertex.attributes['X'].value) / NewFact))
                                y = int(round(float(vertex.attributes['Y'].value) / NewFact))
                                xy_neg[regionID].append((x, y))

        img = Image.new('L', (int(ImgMaxSizeX_orig / NewFact), int(ImgMaxSizeY_orig / NewFact)), 0)
        for regionID in xy.keys():
            xy_a = xy[regionID]
            ImageDraw.Draw(img, 'L').polygon(xy_a, outline=255, fill=255)
        for regionID in xy_neg.keys():
            xy_a = xy_neg[regionID]
            ImageDraw.Draw(img, 'L').polygon(xy_a, outline=255, fill=0)

        mask = np.array(img)

        if Attribute_Name == "non_selected_regions":
            scipy.misc.toimage(255 - mask).save(os.path.join(os.path.split(self._basename[:-1])[0],
                                                             "mask_" + os.path.basename(
                                                                 self._basename) + "_" + Attribute_Name + ".jpeg"))
        else:
            if self._mask_type == 0:
                scipy.misc.toimage(255 - mask).save(os.path.join(os.path.split(self._basename[:-1])[0],
                                                                 "mask_" + os.path.basename(
                                                                     self._basename) + "_" + Attribute_Name + "_inv.jpeg"))
            else:
                scipy.misc.toimage(mask).save(os.path.join(os.path.split(self._basename[:-1])[0],
                                                           "mask_" + os.path.basename(
                                                               self._basename) + "_" + Attribute_Name + ".jpeg"))

        return mask / 255.0, xml_valid, NewFact
