import imp
import os
from lib.datasets.dataset_catalog import DatasetCatalog
from lib.renderers.urbangiraffe import Renderer as urbangirafferenderer

renderer_dict = {
    # 'bboxNeRF': bboxNeRFtrainer,
    'urbangiraffe': urbangirafferenderer
}


def _renderer_factory(cfg):
    renderer = renderer_dict[cfg.renderer_type](cfg)
    return renderer

def make_renderer(cfg):
        return _renderer_factory(cfg)