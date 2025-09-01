import os
from PIL import Image
from multiprocessing import Process, JoinableQueue
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
from imageio import imsave
import numpy as np
from skimage import io, color

os.add_dll_directory(r"D:\TMB\WSI切分\svsCut\openslide_dll\openslide-bin-4.0.0.3-windows-x64\bin\libopenslide-1.dll")

class TileWorker(Process):
    """
    一个子进程，用于生成和保存瓦片图像。

    Args:
        queue (JoinableQueue): 用于进程间通信的队列。
        slidepath (str): SVS文件的路径。
        tile_size (tuple): 瓦片的大小，格式为(宽度, 高度)。
        overlap (int): 瓦片之间的重叠像素数。
        limit_bounds (bool): 是否限制瓦片的边界。
        quality (int): 图像保存的质量（通常用于JPEG格式）。
        _Bkg (float): 背景阈值百分比，用于判断瓦片是否主要是背景。
        _ROIpc (float): ROI（感兴趣区域）阈值百分比，用于判断瓦片是否包含足够的ROI信息。

    Attributes:
        _queue (JoinableQueue): 用于进程间通信的队列。
        _slidepath (str): SVS文件的路径。
        _tile_size (tuple): 瓦片的大小。
        _overlap (int): 瓦片之间的重叠像素数。
        _limit_bounds (bool): 是否限制瓦片的边界。
        _quality (int): 图像保存的质量。
        _slide (OpenSlide): OpenSlide对象，用于读取SVS文件。
        _Bkg (float): 背景阈值百分比。
        _ROIpc (float): ROI阈值百分比。

    Methods:
        RGB_to_lab(tile): 将RGB图像转换为Lab颜色空间。
        Lab_to_RGB(Lab): 将Lab颜色空间的图像转换回RGB。
        normalize_tile(tile, NormVec): 对瓦片进行归一化处理。
        run(): 子进程的主运行函数，从队列中获取任务并执行。
        _get_dz(associated=None): 获取DeepZoomGenerator对象，用于生成瓦片。
    """

    def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, quality, _Bkg, _ROIpc):
        # 初始化一个进程名为TileWorker
        Process.__init__(self, name='TileWorker')
        self.daemon = True
        # 用于进程通信的队列
        self._queue = queue
        # svs文件所在路径
        self._slidepath = slidepath
        # 瓦片的尺寸
        self._tile_size = tile_size
        # 瓦片之间重叠的像素数
        self._overlap = overlap
        # 是否限制瓦片边界
        self._limit_bounds = limit_bounds
        # 图像保存质量
        self._quality = quality
        # OpenSlide对象，初始化为None后在run方法中加载
        self._slide = None
        # 背景阈值百分比
        self._Bkg = _Bkg
        # ROI阈值百分比
        self._ROIpc = _ROIpc

    def RGB_to_lab(self, tile):
        # 使用skimage库将RGB图像转换为Lab颜色空间
        Lab = color.rgb2lab(tile)
        return Lab

    def Lab_to_RGB(self, Lab):
        # 将Lab颜色空间转换回RGB，并缩放到0-255范围，转换为uint8
        newtile = (color.lab2rgb(Lab) * 255).astype(np.uint8)
        return newtile

    def normalize_tile(self, tile, NormVec):
        # 将RGB图像转换为Lab颜色空间
        Lab = self.RGB_to_lab(tile)
        TileMean = [0, 0, 0]
        TileStd = [1, 1, 1]
        newMean = NormVec[0:3]
        newStd = NormVec[3:6]
        # 计算原始瓦片的Lab颜色空间的均值和标准差
        for i in range(3):
            TileMean[i] = np.mean(Lab[:, :, i])
            TileStd[i] = np.std(Lab[:, :, i])
            print("mean/std chanel " + str(i) + ": " + str(TileMean[i]) + " / " + str(TileStd[i]))
            Lab[:, :, i] = ((Lab[:, :, i] - TileMean[i]) * (newStd[i] / TileStd[i])) + newMean[i]
        tile = self.Lab_to_RGB(Lab)
        return tile

    def process_tile(self, tile, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask,
                     Normalize):
        gray = tile.convert('L')
        bw = gray.point(lambda x: 0 if x < 220 else 1, '1')
        arr = np.array(bw)
        avgBkg = np.average(bw)

        if avgBkg <= (self._Bkg / 100.0) and PercentMasked >= (self._ROIpc / 100.0):
            if Normalize:
                tile = Image.fromarray(self.normalize_tile(tile, Normalize).astype('uint8'), 'RGB')
            tile.save(outfile, quality=self._quality)

            if SaveMasks:
                height, width = TileMask.shape
                TileMaskO = np.zeros((height, width), 'uint8')
                maxVal = float(TileMask.max())
                TileMaskO = (TileMask.astype(float) / maxVal * 255.0).astype(int)
                TileMaskO[TileMaskO < 10] = 0
                TileMaskO[TileMaskO >= 10] = 255
                TileMaskO_img = Image.fromarray(TileMaskO)
                TileMaskO_img = TileMaskO_img.resize((arr.shape[0], arr.shape[1]), Image.ANTIALIAS)
                imsave(outfile_bw, TileMaskO_img)

    def run(self):
        self._slide = open_slide(self._slidepath)
        last_associated = None
        dz = self._get_dz()
        while True:
            data = self._queue.get()
            if data is None:
                self._queue.task_done()
                break

            associated, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask, Normalize = data
            if last_associated != associated:
                dz = self._get_dz(associated)
                last_associated = associated

            try:
                tile = dz.get_tile(level, address)
                self.process_tile(tile, level, address, outfile, format, outfile_bw, PercentMasked, SaveMasks, TileMask,
                                  Normalize)
                self._queue.task_done()
            except Exception as e:
                print(f"image {self._slidepath} failed at dz.get_tile for level {level}: {e}")
                self._queue.task_done()

    def _get_dz(self, associated=None):
        if associated is not None:
            image = ImageSlide(self._slide.associated_images[associated])
        else:
            image = self._slide
        return DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)