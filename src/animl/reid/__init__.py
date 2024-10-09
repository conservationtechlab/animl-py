from animl.reid import heads
from animl.reid import miewid
from animl.reid import viewpoint

from animl.reid.heads import (ArcFaceLossAdaptiveMargin,
                              ArcFaceSubCenterDynamic, ArcMarginProduct,
                              ArcMarginProduct_subcenter, ElasticArcFace,
                              l2_norm,)
from animl.reid.miewid import (GeM, IMAGE_HEIGHT, IMAGE_WIDTH, MiewIdNet, load,
                               matchypatchy, weights_init_classifier,
                               weights_init_kaiming,)
from animl.reid.viewpoint import (IMAGE_HEIGHT, IMAGE_WIDTH, ViewpointModel,
                                  load, matchypatchy,)

__all__ = ['ArcFaceLossAdaptiveMargin', 'ArcFaceSubCenterDynamic',
           'ArcMarginProduct', 'ArcMarginProduct_subcenter', 'ElasticArcFace',
           'GeM', 'IMAGE_HEIGHT', 'IMAGE_WIDTH', 'MiewIdNet', 'ViewpointModel',
           'heads', 'l2_norm', 'load', 'matchypatchy', 'miewid', 'viewpoint',
           'weights_init_classifier', 'weights_init_kaiming']
