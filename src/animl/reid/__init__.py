from animl.reid import distance
from animl.reid import inference
from animl.reid import miewid

from animl.reid.distance import (compute_batched_distance_matrix,
                                 compute_distance_matrix, cosine_distance,
                                 euclidean_squared_distance, remove_diagonal,)
from animl.reid.inference import (extract_miew_embeddings, load_miew,)
from animl.reid.miewid import (ArcFaceLossAdaptiveMargin,
                               ArcFaceSubCenterDynamic, ArcMarginProduct,
                               ArcMarginProduct_subcenter, ElasticArcFace, GeM,
                               MIEWID_SIZE, MiewIdNet, l2_norm,)

__all__ = ['ArcFaceLossAdaptiveMargin', 'ArcFaceSubCenterDynamic',
           'ArcMarginProduct', 'ArcMarginProduct_subcenter', 'ElasticArcFace',
           'GeM', 'MIEWID_SIZE', 'MiewIdNet',
           'compute_batched_distance_matrix', 'compute_distance_matrix',
           'cosine_distance', 'distance', 'euclidean_squared_distance',
           'extract_miew_embeddings', 'inference', 'l2_norm', 'load_miew',
           'miewid', 'remove_diagonal']
