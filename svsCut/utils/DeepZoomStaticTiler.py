from __future__ import print_function
import json
import os
import xml.etree.ElementTree as ET
import jinja2

# 设置openslide二进制文件的路径
os.add_dll_directory(r'D:\TMB\WSI切分\svsCut\openslide_dll\openslide-bin-4.0.0.3-windows-x64\bin')
import openslide
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator

import re
import shutil
from unicodedata import normalize
import subprocess

from multiprocessing import JoinableQueue
from PIL import Image

from utils.DeepZoomImageTiler import DeepZoomImageTiler
from utils.TileWorker import TileWorker

Image.MAX_IMAGE_PIXELS = None

VIEWER_SLIDE_NAME = 'slide'


class DeepZoomStaticTiler(object):
    """Handles generation of tiles and metadata for all images in a slide."""

    def __init__(self, slidepath, basename, format, tile_size, overlap,
                 limit_bounds, quality, workers, with_viewer, Bkg, basenameJPG, xmlfile, mask_type, ROIpc, oLabel,
                 ImgExtension, SaveMasks, Mag, normalize):
        if with_viewer:
            # Check extra dependency before doing a bunch of work
            import jinja2
        self._slide = open_slide(slidepath)
        self._basename = basename
        self._basenameJPG = basenameJPG
        self._xmlfile = xmlfile
        self._mask_type = mask_type
        self._format = format
        self._tile_size = tile_size
        self._overlap = overlap
        self._limit_bounds = limit_bounds
        self._queue = JoinableQueue(2 * workers)
        self._workers = workers
        self._with_viewer = with_viewer
        self._Bkg = Bkg
        self._ROIpc = ROIpc
        self._dzi_data = {}
        self._xmlLabel = oLabel
        self._ImgExtension = ImgExtension
        self._SaveMasks = SaveMasks
        self._Mag = Mag
        self._normalize = normalize

        for _i in range(workers):
            TileWorker(self._queue, slidepath, tile_size, overlap,
                       limit_bounds, quality, self._Bkg, self._ROIpc).start()

    def run(self):
        self._run_image()
        if self._with_viewer:
            for name in self._slide.associated_images:
                self._run_image(name)
            self._write_html()
            self._write_static()
        self._shutdown()

    def _run_image(self, associated=None):
        """Run a single image from self._slide."""
        if associated is None:
            image = self._slide
            if self._with_viewer:
                basename = os.path.join(self._basename, VIEWER_SLIDE_NAME)
            else:
                basename = self._basename
        else:
            image = ImageSlide(self._slide.associated_images[associated])
            basename = os.path.join(self._basename, self._slugify(associated))
        dz = DeepZoomGenerator(image, self._tile_size, self._overlap, limit_bounds=self._limit_bounds)
        tiler = DeepZoomImageTiler(dz, basename, self._format, associated, self._queue, self._slide, self._basenameJPG,
                                   self._xmlfile, self._mask_type, self._xmlLabel, self._ROIpc, self._ImgExtension,
                                   self._SaveMasks, self._Mag, self._normalize)
        tiler.run()
        self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

    def _url_for(self, associated):
        if associated is None:
            base = VIEWER_SLIDE_NAME
        else:
            base = self._slugify(associated)
        return '%s.dzi' % base

    def _write_html(self):

        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__), autoescape=True)
        template = env.get_template('slide-multipane.html')
        associated_urls = dict((n, self._url_for(n))
                               for n in self._slide.associated_images)
        try:
            mpp_x = self._slide.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = self._slide.properties[openslide.PROPERTY_NAME_MPP_Y]
            mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            mpp = 0

        data = template.render(slide_url=self._url_for(None), slide_mpp=mpp, associated=associated_urls,
                               properties=self._slide.properties, dzi_data=json.dumps(self._dzi_data))
        with open(os.path.join(self._basename, 'index.html'), 'w') as fh:
            fh.write(data)

    def _write_static(self):
        basesrc = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'static')
        basedst = os.path.join(self._basename, 'static')
        self._copydir(basesrc, basedst)
        self._copydir(os.path.join(basesrc, 'images'),
                      os.path.join(basedst, 'images'))

    def _copydir(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for name in os.listdir(src):
            srcpath = os.path.join(src, name)
            if os.path.isfile(srcpath):
                shutil.copy(srcpath, os.path.join(dest, name))

    @classmethod
    def _slugify(cls, text):
        text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
        return re.sub('[^a-z0-9]+', '_', text)

    def _shutdown(self):
        for _i in range(self._workers):
            self._queue.put(None)
        self._queue.join()


def ImgWorker(queue):
    while True:
        cmd = queue.get()
        if cmd is None:
            queue.task_done()
            break
        subprocess.Popen(cmd, shell=True).wait()
        queue.task_done()


def xml_read_labels(xmldir):
    try:
        # 使用 ElementTree 解析 XML 文件
        tree = ET.parse(xmldir)
        root = tree.getroot()
        xml_valid = True
    except ET.ParseError as e:
        # 捕获 XML 解析错误
        print(f"Error parsing XML file: {e}")
        return [], False
    except Exception as e:
        # 捕获其他可能发生的异常
        print(f"Unexpected error occurred: {e}")
        return [], False

    xml_labels = []
    # 遍历所有的 <Attribute> 标签
    for attr in root.findall('.//Attribute'):
        # 检查 'Value' 属性是否存在
        if 'Value' in attr.attrib:
            xml_labels.append(attr.attrib['Value'])
        else:
            print("Warning: 'Value' attribute missing in an <Attribute> tag.")

    # 如果列表为空，返回一个包含空字符串的列表
    if not xml_labels:
        xml_labels = ['']

    return xml_labels, xml_valid
