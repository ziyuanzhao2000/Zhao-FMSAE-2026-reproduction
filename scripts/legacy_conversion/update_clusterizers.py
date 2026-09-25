from fmsae_repro import MODELS, load_manifest
import os
import joblib
import ezslide
import lazyslide as zs
import mesoslide as ms
import pandas as pd
from glob import glob
from tqdm import tqdm 

OLD_CLUSTERIZERS_DIR = f"{MODELS}/token_clusterers/legacy"
old_clusterizer_paths = glob(os.path.join(OLD_CLUSTERIZERS_DIR, "UNI_SAE_*.joblib"))
old_clusterizer_paths.sort()

manifest = load_manifest()
slides = ezslide.open_slides(manifest, attach_images=False)

from lazyslide_models import MODEL_REGISTRY

model_module = MODEL_REGISTRY["uni"]
uni = model_module()  # Initiate the model

for clusterizer_path in tqdm(old_clusterizer_paths):
    feature_name = os.path.basename(clusterizer_path).replace(".joblib", "")
    print(f"Processing {feature_name}...")
    output_path = f'{MODELS}/token_clusterers/updated/{feature_name}.joblib'    
    if os.path.exists(output_path):
        continue    
    clusterizer = joblib.load(clusterizer_path)
    clusterizer.device = 'cuda:1'
    clusterizer.fit(slides, model=uni, feature_name=feature_name)
    clusterizer.device = 'cpu'
    joblib.dump(clusterizer, output_path)
