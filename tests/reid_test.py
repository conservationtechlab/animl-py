import unittest
import time
import numpy as np
from pathlib import Path


import animl
from animl.utils.general import NUM_THREADS


# NOTE: DIFFERENT OUTPUT ON SECOND RUN, MAYBE RELATED TO SEED?

@unittest.skip
def reid_test():
    start_time = time.time()

    manifest_path = Path.cwd() / 'examples' / 'Jaguar' / 'reid_manifest.csv'
    miew_path = Path.cwd() / 'models/miewid_v3.bin'
    manifest = animl.load_data(manifest_path)

    miew = animl.load_miew(miew_path)
    embeddings = animl.extract_miew_embeddings(miew,
                                               manifest,
                                               file_col="filepath",
                                               batch_size=4,
                                               num_workers=NUM_THREADS)

    e2 = animl.compute_distance_matrix(embeddings, embeddings, metric='euclidean')
    cos = animl.compute_distance_matrix(embeddings, embeddings, metric='cosine')

    e2_gt_filepath = Path.cwd() / 'tests' / 'GroundTruth' / 'reid' / 'e2.npy'
    cos_gt_filepath = Path.cwd() / 'tests' / 'GroundTruth' / 'reid' / 'cos.npy'
    e2_gt = np.load(e2_gt_filepath)
    cos_gt = np.load(cos_gt_filepath)

    e2_match = (e2 == e2_gt).all()
    cos_match = (cos == cos_gt).all()

    batched = animl.compute_batched_distance_matrix(embeddings, embeddings, metric='cosine', batch_size=2)

    batched_filepath = Path.cwd() / 'tests' / 'GroundTruth' / 'reid' / 'batched.npy'
    batched_gt = np.load(batched_filepath)
    batched_match = (batched == batched_gt).all()

    print(e2_match, cos_match, batched_match)
    if e2_match and cos_match and batched_match:
        print("ReID Test Successful!")
    else:
        print(e2==e2_gt)
        print(e2_gt)
    print(f"Test completed in {time.time() - start_time:.2f} seconds")

    #TODO : remove_diagonal()

reid_test()