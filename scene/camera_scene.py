import os
from scene.dataset_readers import readColmapSceneInfo, readColmapSceneInfo_hw
from utils.camera_utils import cameraList_load

class CamScene:
    def __init__(self, source_path, h=512, w=512, aspect=-1):
        """b
        :param path: Path to colmap scene main folder.
        """
        if aspect != -1:
            h = 512
            w = 512 * aspect
        print(f'source_path is {source_path},h is {h}, w is {w}, aspect is {aspect}')
        if os.path.exists(os.path.join(source_path, "sparse")):
            if h == -1 or w == -1:
                scene_info = readColmapSceneInfo(source_path, None, False)
                h = scene_info.train_cameras[0].height
                w = scene_info.train_cameras[0].width
                if w > 1920:
                    scale = w / 1920
                    h /= scale
                    w /= scale
            else:
                scene_info = readColmapSceneInfo_hw(source_path, h, w, None, False)

        else:
            assert False, "Could not recognize scene type!"

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        print(f'scene_info.train_cameras[0] is {scene_info.train_cameras[0]}')
        self.cameras = cameraList_load(scene_info.train_cameras, h, w)