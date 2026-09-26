from fmsae_repro import DATA, RAW_DATA
import os
import sys
from glob import glob
import ezslide
import lazyslide as zs
import mesoslide as ms


def main():
    file_paths = glob(f'{RAW_DATA}/3D_HnE/*.ome.tif')
    file_paths.sort()

    file_path = file_paths[int(sys.argv[1])]
    lsp_id = os.path.basename(file_path).split('.')[0]
    print(f'Processing {lsp_id} from {file_path}')

    wsi = ezslide.open_slide(file_path,
                           attach_images=True,
                           reader="tifffile_zarr")
    zs.pp.find_tissues(wsi,
                       detect_holes=False)
    zs.pp.tile_tissues(wsi,
                       tile_px=448,
                       stride_px=256,
                       background_filter=False)
    wsi = ms.tl.feature_extraction(
        wsi,
        model='uni',
        tile_key='tiles',
        key_added='UNI_embedding',
        token=os.environ.get('HF_TOKEN'),
        batch_size=128,
        device='cuda:0',
        num_workers=1,
        save=False,  # don't persist back to the zarr store, this is just a comparison
    )
    wsi.write(f'{DATA}/{lsp_id}.zarr')


if __name__ == '__main__':
    main()