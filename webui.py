import random
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torchvision
from argparse import ArgumentParser
import viser
import viser.transforms as tf
from scene import GaussianModel
from arguments import PipelineParams, ModelParams
from scene.camera_scene import CamScene
from scene.cameras import Simple_Camera
from scene import Scene, SimpleScene
from segment_anything import sam_model_registry, SamPredictor
from gaussian_renderer import render
import math
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from simple_knn._C import distCUDA10
from lang_sam import LangSAM
from PIL import Image

class WebUI:
    def __init__(self, cfg) -> None:
        self.model_path = cfg.gs_source
        self.colmap_dir = cfg.colmap_path
        self.port = 8084
        self.server = viser.ViserServer(port=self.port)

        self.colmap_cameras = None
        self.render_cameras = None
        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        self.gaussian = GaussianModel(sh_degree=3, category_num=32)

        self.gaussian.load_ply(self.model_path)
        self.background_tensor = torch.tensor(
            [1, 1, 1], dtype=torch.float32, device="cuda"
        )

        if self.colmap_dir is not None:
            self.scene = SimpleScene(self.colmap_dir)
            self.camera_extent = self.scene.camera_extent
            self.colmap_cameras = self.scene.cameras

        with self.server.add_gui_folder("operator"):
            self.reset_scene = self.server.add_gui_button("Reset scene")

        with self.server.add_gui_folder("Point prompt"):
            self.left_up = self.server.add_gui_vector2(
                "Left UP",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )
            self.begin_seg = self.server.add_gui_checkbox(
                "Input point prompt", initial_value=False
            )
            self.segmentation = self.server.add_gui_button(
                "Segment", visible=True
            )
            self.point_seg_time = self.server.add_gui_text(
                "Segmentation time: ",
                initial_value="",
                visible=False
            )

        with self.server.add_gui_folder("Multi-point prompt"):
            self.click_point = self.server.add_gui_vector2(
                "click_point",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )
            self.begin_multi_prompt = self.server.add_gui_checkbox(
                "Multi-point prompt", initial_value=False
            )
            self.operator = self.server.add_gui_dropdown("Click Operator", ["Add point", "Delete Point"])

            self.multi_point_segmentation = self.server.add_gui_button(
                "Segment", visible=True
            )
            self.multi_seg_time = self.server.add_gui_text(
                "Segmentation time: ",
                initial_value="",
                visible=False
            )

        self.click_points = []
        self.click_type = []


        with self.server.add_gui_folder("Text prompt"):
            self.text_prompt = self.server.add_gui_text(
                "Text prompt",
                initial_value="",
                visible=True
            )
            self.text_segment = self.server.add_gui_button(
                "Segment", visible=True
            )
            self.text_seg_time = self.server.add_gui_text(
                "Segmentation time: ",
                initial_value="",
                visible=False
            )

        with self.server.add_gui_folder("Save"):
            self.save_ply = self.server.add_gui_button(
                "Save Ply", visible=True
            )
            self.save_image = self.server.add_gui_button(
                "Save image", visible = True
            )
            self.ply_save_path = self.server.add_gui_text(
                "Ply save path",
                initial_value="",
                visible=False
            )

        with torch.no_grad():
            self.frames = []
            random.seed(0)
            # frame_index = random.sample(
            #     range(0, len(self.colmap_cameras)),
            #     min(len(self.colmap_cameras), 40)
            # )
            frame_index = [0]
            for i in frame_index:
                self.make_one_camera_pos_frame(i)


        # print("开始加载SAM模型")
        self.sam_checkpoint = "Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda"
        self.sam_predictor = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam_predictor.to(device=self.device)
        self.predictor = SamPredictor(self.sam_predictor)
        print("SAM模型加载完成")

        # print("开始加载LangSAM模型")
        # self.lang_model = LangSAM(self.model_type, self.sam_checkpoint)
        # print("LangSAM模型加载完成")

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(32, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 128, bias=True),
            torch.nn.LayerNorm(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128, bias=True),
            torch.nn.LayerNorm(128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 256, bias=True),
        )
        self.classifier.cuda()
        self.classifier.load_state_dict(torch.load(cfg.pth_path))
        self.classifier.eval()

        self.has_seg = False


        @self.reset_scene.on_click
        def _(event: viser.GuiEvent):
            # self.gaussian = GaussianModel(sh_degree=3, category_num=256)
            self.gaussian.load_ply(self.model_path)


        @self.save_ply.on_click
        def _(event: viser.GuiEvent):
            self.ply_save_path.value = "The PLY file has been saved to the folder" + os.path.abspath(os.getcwd())
            self.ply_save_path.visible = True
            self.gaussian.save_object_ply(os.path.join('data', 'caijian', 'point_cloud.ply'))
            data_path =os.path.join('data', 'caijian', 'output.ply')
            print(f'output has saved in {data_path}')


        @self.text_segment.on_click
        def _(event: viser.GuiEvent):
            img = np.asarray(to_pil_image(self.render_pkg["render"].cpu()))
            pil_img = Image.fromarray(np.uint8(img)).convert("RGB")
            print(f'self.text_prompt.value is {self.text_prompt.value}')
            masks, boxes, phrases, logits = self.lang_model.predict(pil_img, self.text_prompt.value)
            print(f'masks.shape is {masks.shape}')
            self.render_pkg["sam_mask"] = []
            self.render_pkg["sam_mask"].append(masks[0])
            start_time = time.time()
            image_category = self.render_pkg["render_category"].permute(1, 2, 0).view(-1, 32)
            logits = self.classifier(image_category)
            predic = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            mask = self.render_pkg["sam_mask"][0].flatten()
            predic = predic[mask]
            exclude_value = 10
            # 计算每个值的出现次数
            counts = torch.bincount(predic)
            # 将指定值的计数设为0
            counts[exclude_value] = 0
            # 找到出现次数最多的值
            most_common_value = torch.argmax(counts)
            print(f'出现最多的是{most_common_value.item()}')

            with torch.no_grad():
                logits_3d = self.classifier(self.gaussian._features_category)

            predictions = torch.argmax(torch.softmax(logits_3d, dim=1), dim=1)

            target_gaussian = predictions == most_common_value.item()
            self.gaussian.delete_points(target_gaussian)
            #
            # dist2 = torch.clamp_min(distCUDA10(self.gaussian.get_xyz).cuda(), 0.0000001)
            # dist_mask = dist2 < self.scene.camera_extent / 200
            # self.gaussian.delete_points(dist_mask)

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.text_seg_time.value = f'{elapsed_time:.2f}s'
            self.text_seg_time.visible = True


        @self.server.on_scene_click
        def _(pointer):
            self.click_cb(pointer)


        @self.multi_point_segmentation.on_click
        def _(event: viser.GuiEvent):
            start_time = time.time()
            image_category = self.render_pkg["render_category"].permute(1, 2, 0).view(-1, 32)
            logits = self.classifier(image_category)
            predic = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            mask = self.render_pkg["sam_mask"][0].flatten()
            predic = predic[mask]
            exclude_value = 0
            # 计算每个值的出现次数
            counts = torch.bincount(predic)
            # 将指定值的计数设为0
            counts[exclude_value] = 0
            # 找到出现次数前N个的值
            array = torch.tensor(self.click_type)
            N = torch.sum(array == 1).item() - torch.sum(array == 0).item()
            top_n_values = torch.topk(counts, N).indices
            with torch.no_grad():
                logits_3d = self.classifier(self.gaussian._features_category)

            predictions = torch.argmax(torch.softmax(logits_3d, dim=1), dim=1)

            # Create a mask to identify points to delete
            target_gaussian = torch.zeros_like(predictions, dtype=torch.bool)
            for value in top_n_values:
                print(value.item())
                target_gaussian |= (predictions == value.item())

            self.gaussian.delete_points(target_gaussian)

            # dist2 = torch.clamp_min(distCUDA10(self.gaussian.get_xyz).cuda(), 0.0000001)
            # dist_mask = dist2 < self.scene.camera_extent / 2000
            # self.gaussian.delete_points(dist_mask)

            end_time = time.time()
            elapsed_time = end_time - start_time
            self.point_seg_time.value = f'{elapsed_time:.2f}s'
            self.point_seg_time.visible = True
            self.has_seg = True

        @self.save_image.on_click
        def _(event:viser.GuiEvent):
            # 假设 render_image 是你的张量数据
            image = self.prepare_out_image(self.render_pkg)
            image = Image.fromarray(image, mode='RGB')
            image.save('output_image.png')
            print(f'图片已经保存到"output_image.png"')


        @self.segmentation.on_click
        def _(event: viser.GuiEvent):
            start_time = time.time()
            image_category = self.render_pkg["render_category"].permute(1, 2, 0).view(-1, 32)
            logits = self.classifier(image_category)
            predic = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            mask = self.render_pkg["sam_mask"][0].flatten()
            predic = predic[mask]
            # exclude_value = 193
            # 计算每个值的出现次数
            counts = torch.bincount(predic)
            # 将指定值的计数设为0
            # counts[exclude_value] = 0
            # 找到出现次数最多的值
            most_common_value = torch.argmax(counts)
            print(f'出现最多的是{most_common_value.item()}')
            with torch.no_grad():
                logits_3d = self.classifier(self.gaussian.get_category)

            predictions = torch.argmax(torch.softmax(logits_3d, dim=1), dim=1)

            target_gaussian = predictions == most_common_value.item()
            self.gaussian.delete_points(target_gaussian)

            dist2 = torch.clamp_min(distCUDA10(self.gaussian.get_xyz).cuda(), 0.0000001)
            dist_mask = dist2 < self.scene.camera_extent / 100
            self.gaussian.delete_points(dist_mask)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.point_seg_time.value = f'{elapsed_time:.2f}s'
            self.point_seg_time.visible = True
            self.has_seg = True


    def click_cb(self, pointer):
        click_pos = pointer.click_pos
        click_pos = torch.tensor(click_pos)
        cur_cam = self.camera
        self.left_up.value = [
            int(cur_cam.image_width * click_pos[0]),
            int(cur_cam.image_height * click_pos[1])
        ]
        # 如果进行多点添加
        if self.begin_multi_prompt.value:
            # 如果当前选择是添加点
            if self.operator.value == 'Add point':
                self.click_points.append([self.left_up.value[0], self.left_up.value[1]])
                self.click_type.append(1)
            if self.operator.value == "Delete Point":
                self.click_points.append([self.left_up.value[0], self.left_up.value[1]])
                self.click_type.append(0)

    def make_one_camera_pos_frame(self, idx):
        cam = self.colmap_cameras[idx]
        print(f'cam is {cam.uid}, {cam.image_name}')
        T_world_camera = tf.SE3.from_rotation_and_translation(
            tf.SO3(cam.qvec), cam.T
        ).inverse()

        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()
        frame = self.server.add_frame(
            f'/colmap/frame_{idx}',
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=True
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            print("frame.on_click")
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):
            def begin_trans(client):
                assert client is not None
                # 当前的旋转，缩放坐标
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
                # 目标frame的旋转，缩放
                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target
                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position
            # 将视角放到点击的frame那里
            self.begin_call = begin_trans

    @torch.no_grad()
    def sam_predict(self, image):
        img = np.asarray(to_pil_image(image.cpu()))
        self.predictor.set_image(img)
        _mask, _, _ = self.predictor.predict(
            point_coords=np.array([[self.left_up.value[0], self.left_up.value[1]]]),
            point_labels=np.array([1]),
            multimask_output=False
        )
        _mask = torch.from_numpy(_mask).to(torch.bool).cuda()
        return _mask.squeeze()

    @torch.no_grad()
    def multi_sam_predict(self, image):
        img = np.asarray(to_pil_image(image.cpu()))
        self.predictor.set_image(img)
        print(f'self.click_points: {self.click_points}')
        print(f'self.click_type: {self.click_type}')
        _mask, _, _ = self.predictor.predict(
            point_coords=np.array(self.click_points),
            point_labels=np.array(self.click_type),
            multimask_output=False
        )
        _mask = torch.from_numpy(_mask).to(torch.bool).cuda()
        return _mask.squeeze()

    def render_loop(self):
        while True:
            self.update_viewer()
            time.sleep(0.1)

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        if self.render_cameras is None and self.colmap_dir is not None:
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
            self.render_cameras = SimpleScene(self.colmap_dir).cameras
            self.begin_call(list(self.server.get_clients().values())[0])


        viser_cam = list(self.server.get_clients().values())[0].camera
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        fovy = self.render_cameras[0].FoVy
        fovx = 2 * math.atan(math.tan(fovy / 2))
        width = self.render_cameras[0].image_width
        height = self.render_cameras[0].image_height
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def render(self, cam):
        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii, render_category = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["render_category"]
        )
        self.render_pkg = render_pkg
        render_pkg["sam_mask"] = []
        if self.begin_seg.value and not self.has_seg:
            sam_output = self.sam_predict(image)
            if sam_output is not None:
                render_pkg["sam_mask"].append(sam_output)

        if self.begin_multi_prompt.value and not self.has_seg and len(self.click_points) > 0:
            sam_output = self.multi_sam_predict(image)
            if sam_output is not None:
                render_pkg["sam_mask"].append(sam_output)

        # if self.add_point.value or self.del_point.value:

        return render_pkg

    @torch.no_grad()
    def prepare_out_image(self, render_pkg):
        render_image = render_pkg["render"]
        out = render_image.clamp(0, 1)
        out = (out * 255).to(torch.uint8).cpu().to(torch.uint8)
        _mask = render_pkg["sam_mask"]
        if len(_mask) > 0:
            out = torchvision.utils.draw_segmentation_masks(
                out, _mask[0]
            )
        return out.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            print("---------------------------camera is None !!! ---------------------------")
            return
        render_pkg = self.render(gs_camera)
        image = self.prepare_out_image(render_pkg)
        self.server.set_background_image(image, format="jpeg")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)
    # colmap_dir: colmap计算出的文件路径
    parser.add_argument("--colmap_path", type=str, required=True)  #
    parser.add_argument("--pth_path", type=str, required=True)
    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()