import base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch

from scripts.sketch_helper import get_high_freq_colors, color_quantization, create_binary_matrix_base64, create_binary_mask
import numpy as np
import cv2

from modules import devices, script_callbacks

import modules.scripts as scripts
import gradio as gr

from modules.script_callbacks import CFGDenoisedParams, on_cfg_denoised

from modules.processing import StableDiffusionProcessing

MAX_COLORS = 12
switch_values_symbol = '\U000021C5' # ⇅


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


# abstract base class for filters
from abc import ABC, abstractmethod




class Filter(ABC):

    @abstractmethod
    def create_tensor(self):
        pass



@dataclass
class Division:
    y: float
    x: float


@dataclass
class Position:
    y: float
    x: float
    ey: float
    ex: float



class RectFilter(Filter):
    def __init__(self, division: Division, position: Position, weight: float):
        self.division = division
        self.position = position
        self.weight = weight

    def create_tensor(self, num_channels: int, height_b: int, width_b: int) -> torch.Tensor:

        x = torch.zeros(num_channels, height_b, width_b).to(devices.device)

        division_height = height_b / self.division.y
        division_width = width_b / self.division.x
        y1 = int(division_height * self.position.y)
        y2 = int(division_height * self.position.ey)
        x1 = int(division_width * self.position.x)
        x2 = int(division_width * self.position.ex)

        x[:, y1:y2, x1:x2] = self.weight

        return x


class MaskFilter:
    def __init__(self, binary_mask: np.array = None, weight: float = None, float_mask: np.array = None):
        if float_mask is None:
            self.mask = binary_mask.astype(np.float32) * weight
        elif binary_mask is None and weight is None:
            self.mask = float_mask
        else:
            raise ValueError('Either float_mask or binary_mask and weight must be provided')
        self.tensor_mask = torch.tensor(self.mask).to(devices.device)

    def create_tensor(self, num_channels: int, height_b: int, width_b: int) -> torch.Tensor:


        # x = torch.zeros(num_channels, height_b, width_b).to(devices.device)
        # mask = torch.tensor(self.mask).to(devices.device)
        # downsample mask to x size
        # mask_bicubic = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height_b, width_b), mode='bicubic').squeeze(0).squeeze(0).cpu().numpy()
        #
        # mask_nearest_exact = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height_b, width_b), mode='nearest-exact').squeeze(0).squeeze(0).cpu().numpy()
        #
        # mask_nearest = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height_b, width_b), mode='nearest').squeeze(0).squeeze(0).cpu().numpy()
        #
        # mask_area = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(height_b, width_b), mode='area').squeeze(0).squeeze(0).cpu().numpy()

        mask = torch.nn.functional.interpolate(self.tensor_mask.unsqueeze(0).unsqueeze(0), size=(height_b, width_b), mode='nearest-exact').squeeze(0).squeeze(0)
        mask = mask.unsqueeze(0).repeat(num_channels, 1, 1)

        return mask


class PastePromptTextboxTracker:
    def __init__(self):
        self.scripts = []
        return

    def set_script(self, script):
        self.scripts.append(script)

    def on_after_component_callback(self, component, **_kwargs):

        if not self.scripts:
            return
        if type(component) is gr.State:
            return

        script = None
        if type(component) is gr.Textbox and component.elem_id == 'txt2img_prompt':
            # select corresponding script
            script = next(x for x in self.scripts if x.is_txt2img)
            self.scripts.remove(script)

        if type(component) is gr.Textbox and component.elem_id == 'img2img_prompt':
            # select corresponding script
            script = next(x for x in self.scripts if x.is_img2img)
            self.scripts.remove(script)

        if script is None:
            return

        script.target_paste_prompt = component


prompt_textbox_tracker = PastePromptTextboxTracker()


class Script(scripts.Script):

    def __init__(self):
        self.ui_root = None
        self.num_batches: int = 0
        self.end_at_step: int = 20
        self.filters: List[Filter] = []
        self.debug: bool = False
        self.selected_twoshot_tab = 0
        self.ndmasks = []
        self.area_colors = []
        self.mask_denoise = False
        prompt_textbox_tracker.set_script(self)
        self.target_paste_prompt = None


    def title(self):
        return "Latent Couple extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def create_rect_filters_from_ui_params(self, raw_divisions: str, raw_positions: str, raw_weights: str):

        divisions = []
        for division in raw_divisions.split(','):
            y, x = division.split(':')
            divisions.append(Division(float(y), float(x)))

        def start_and_end_position(raw: str):
            nums = [float(num) for num in raw.split('-')]
            if len(nums) == 1:
                return nums[0], nums[0] + 1.0
            else:
                return nums[0], nums[1]

        positions = []
        for position in raw_positions.split(','):
            y, x = position.split(':')
            y1, y2 = start_and_end_position(y)
            x1, x2 = start_and_end_position(x)
            positions.append(Position(y1, x1, y2, x2))

        weights = []
        for w in raw_weights.split(','):
            weights.append(float(w))

        # todo: assert len

        return [RectFilter(division, position, weight) for division, position, weight in zip(divisions, positions, weights)]

    def create_mask_filters_from_ui_params(self, raw_divisions: str, raw_positions: str, raw_weights: str):

        divisions = []
        for division in raw_divisions.split(','):
            y, x = division.split(':')
            divisions.append(Division(float(y), float(x)))

        def start_and_end_position(raw: str):
            nums = [float(num) for num in raw.split('-')]
            if len(nums) == 1:
                return nums[0], nums[0] + 1.0
            else:
                return nums[0], nums[1]

        positions = []
        for position in raw_positions.split(','):
            y, x = position.split(':')
            y1, y2 = start_and_end_position(y)
            x1, x2 = start_and_end_position(x)
            positions.append(Position(y1, x1, y2, x2))

        weights = []
        for w in raw_weights.split(','):
            weights.append(float(w))

        # todo: assert len

        return [Filter(division, position, weight) for division, position, weight in zip(divisions, positions, weights)]

    def do_visualize(self, raw_divisions: str, raw_positions: str, raw_weights: str):

        self.filters = self.create_rect_filters_from_ui_params(raw_divisions, raw_positions, raw_weights)

        return [f.create_tensor(1, 128, 128).squeeze(dim=0).cpu().numpy() for f in self.filters]

    def do_apply(self, extra_generation_params: str):
        #
        # parse "Latent Couple" extra_generation_params
        #
        raw_params = {}

        for assignment in extra_generation_params.split(' '):
            pair = assignment.split('=', 1)
            if len(pair) != 2:
                continue
            raw_params[pair[0]] = pair[1]

        return raw_params.get('divisions', '1:1,1:2,1:2'), raw_params.get('positions', '0:0,0:0,0:1'), raw_params.get('weights', '0.2,0.8,0.8'), int(raw_params.get('step', '20'))

    def ui(self, is_img2img):
        process_script_params = []
        id_part = "img2img" if is_img2img else "txt2img"
        canvas_html = "<div id='canvas-root' style='max-width:400px; margin: 0 auto'></div>"
        # get_js_colors = """
        # async (canvasData) => {
        #   const canvasEl = document.getElementById("canvas-root");
        #   return [canvasEl._data]
        # }
        # """

        def create_canvas(h, w):
            return np.zeros(shape=(h, w, 3), dtype=np.uint8) + 255

        def process_sketch(img_arr, input_binary_matrixes):
            input_binary_matrixes.clear()
            # base64_img = canvas_data['image']
            # image_data = base64.b64decode(base64_img.split(',')[1])
            # image = Image.open(BytesIO(image_data)).convert("RGB")
            im2arr = img_arr
            # colors = [tuple(map(int, rgb[4:-1].split(','))) for rgb in
            #           ['colors']]
            sketch_colors, color_counts = np.unique(im2arr.reshape(-1, im2arr.shape[2]), axis=0, return_counts=True)
            colors_fixed = []
            # if color count is less than 0.001 of total pixel count, collect it for edge color correction
            edge_color_correction_arr = []
            for sketch_color_idx, color in enumerate(sketch_colors[:-1]):  # exclude white
                if color_counts[sketch_color_idx] < im2arr.shape[0] * im2arr.shape[1] * 0.002:
                    edge_color_correction_arr.append(sketch_color_idx)

            edge_fix_dict = {}
            # TODO:for every non area color pixel in img_arr, find the nearest area color pixel and replace it with that color

            area_colors = np.delete(sketch_colors, edge_color_correction_arr, axis=0)
            if self.mask_denoise:
                for edge_color_idx in edge_color_correction_arr:
                    edge_color = sketch_colors[edge_color_idx]
                    # find the nearest area_color

                    color_distances = np.linalg.norm(area_colors - edge_color, axis=1)
                    nearest_index = np.argmin(color_distances)
                    nearest_color = area_colors[nearest_index]
                    edge_fix_dict[edge_color_idx] = nearest_color
                    # replace edge color with the nearest area_color
                    cur_color_mask = np.all(im2arr == edge_color, axis=2)
                    im2arr[cur_color_mask] = nearest_color

                # recalculate area colors
                sketch_colors, color_counts = np.unique(im2arr.reshape(-1, im2arr.shape[2]), axis=0, return_counts=True)
                area_colors = sketch_colors

            # create binary matrix for each area_color
            area_color_maps = []
            self.ndmasks = []
            self.area_colors = area_colors
            for color in area_colors:
                r, g, b = color
                mask, binary_matrix = create_binary_matrix_base64(im2arr, color)
                self.ndmasks.append(mask)
                input_binary_matrixes.append(binary_matrix)
                colors_fixed.append(gr.update(
                    value=f'<div style="display:flex;justify-content:center;max-height: 94px;"><img width="20%" style="object-fit: contain;flex-grow:1;margin-right: 1em;" src="data:image/png;base64,{binary_matrix}" /><div class="color-bg-item" style="background-color: rgb({r},{g},{b});width:10%;height:auto;"></div></div>'))



            visibilities = []
            sketch_colors = []

            for sketch_color_idx in range(MAX_COLORS):
                visibilities.append(gr.update(visible=False))
                sketch_colors.append(gr.update(value=f'<div class="color-bg-item" style="background-color: black"></div>'))
            for j in range(len(colors_fixed)-1):
                visibilities[j] = gr.update(visible=True)
                sketch_colors[j] = colors_fixed[j]

            alpha_mask_visibility = gr.update(visible=True)
            alpha_mask_html = colors_fixed[-1]
            return [gr.update(visible=True), input_binary_matrixes, alpha_mask_visibility, alpha_mask_html, *visibilities, *sketch_colors]

        def update_mask_filters(alpha_blend_val, general_prompt_str, *cur_weights_and_prompts):
            cur_weight_slider_vals = cur_weights_and_prompts[:MAX_COLORS]
            cur_prompts = cur_weights_and_prompts[MAX_COLORS:]
            general_mask = self.ndmasks[-1]
            final_filter_list = []
            for m in range(len(self.ndmasks) - 1):
                cur_float_mask = self.ndmasks[m].astype(np.float32) * float(cur_weight_slider_vals[m]) * float(1.0-alpha_blend_val)
                mask_filter = MaskFilter(float_mask=cur_float_mask)
                final_filter_list.append(mask_filter)
            # subtract the sum of all masks from the general mask to get the alpha blend mask
            initial_general_mask = np.ones(shape=general_mask.shape, dtype=np.float32)
            alpha_blend_mask = initial_general_mask.astype(np.float32) - np.sum([f.mask for f in final_filter_list], axis=0)
            alpha_blend_filter = MaskFilter(float_mask=alpha_blend_mask)
            final_filter_list.insert(0, alpha_blend_filter)
            self.filters = final_filter_list


            sketch_colors = []
            colors_fixed = []
            for area_idx, color in enumerate(self.area_colors):
                r, g, b = color
                final_list_idx = area_idx + 1
                if final_list_idx == len(final_filter_list):
                    final_list_idx = 0
                # get shape of current mask
                height_b, width_b = final_filter_list[final_list_idx].mask.shape
                current_mask = torch.nn.functional.interpolate(final_filter_list[final_list_idx].tensor_mask.unsqueeze(0).unsqueeze(0),
                                                       size=(int(height_b/8), int(width_b/8)), mode='nearest-exact').squeeze(0).squeeze(0).cpu().numpy()
                adjusted_mask = current_mask * 255
                _, adjusted_mask_arr = cv2.imencode('.png', adjusted_mask)

                adjusted_mask_b64 = base64.b64encode(adjusted_mask_arr.tobytes()).decode('ascii')
                colors_fixed.append(gr.update(
                    value=f'<div style="display:flex;justify-content:center;max-height: 94px;"><img width="20%" style="object-fit: contain;flex-grow:1;margin-right: 1em;" src="data:image/png;base64,{adjusted_mask_b64}" /><div class="color-bg-item" style="background-color: rgb({r},{g},{b});width:10%;height:auto;"></div></div>'))
            for sketch_color_idx in range(MAX_COLORS):

                sketch_colors.append(
                    gr.update(value=f'<div class="color-bg-item" style="background-color: black"></div>'))
            for j in range(len(colors_fixed)-1):

                sketch_colors[j] = colors_fixed[j]
            alpha_mask_visibility = gr.update(visible=True)
            alpha_mask_html = colors_fixed[-1]
            final_prompt_update = gr.update(value='\nAND '.join([general_prompt_str, *cur_prompts[:len(colors_fixed)-1]]))
            return [final_prompt_update, alpha_mask_visibility, alpha_mask_html, *sketch_colors]



        cur_weight_sliders = []

        with gr.Group() as group_two_shot_root:
            binary_matrixes = gr.State([])
            with gr.Accordion("Latent Couple", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                with gr.Tabs(elem_id="script_twoshot_tabs") as twoshot_tabs:

                    with gr.TabItem("Mask", elem_id="tab_twoshot_mask") as twoshot_tab_mask:

                        canvas_data = gr.JSON(value={}, visible=False)
                        # model = gr.Textbox(label="The id of any Hugging Face model in the diffusers format",
                        #                    value="stabilityai/stable-diffusion-2-1-base",
                        #                    visible=False if is_shared_ui else True)
                        mask_denoise_checkbox = gr.Checkbox(value=False, label="Denoise Mask")

                        def update_mask_denoise_flag(flag):
                            self.mask_denoise = flag

                        mask_denoise_checkbox.change(fn=update_mask_denoise_flag, inputs=[mask_denoise_checkbox], outputs=None)
                        canvas_image = gr.Image(source='upload', mirror_webcam=False, type='numpy', tool='color-sketch',
                                                elem_id='twoshot_canvas_sketch', interactive=True).style(height=480)
                        # aspect = gr.Radio(["square", "horizontal", "vertical"], value="square", label="Aspect Ratio",
                        #                   visible=False if is_shared_ui else True)
                        button_run = gr.Button("I've finished my sketch", elem_id="main_button", interactive=True)

                        prompts = []
                        colors = []
                        color_row = [None] * MAX_COLORS
                        with gr.Column(visible=False) as post_sketch:
                            with gr.Row(visible=False) as alpha_mask_row:
                                # general_mask_label_span = gr.HTML(
                                #     '<span class="text-gray-500 text-[0.855rem] mb-2 block dark:text-gray-200 relative z-40">General Mask</span>',
                                #     elem_id='general_mask_label_span')
                                with gr.Box(elem_id="alpha_mask"):
                                    alpha_color = gr.HTML(
                                        '<div class="alpha-mask-item" style="background-color: black"></div>')
                            general_prompt = gr.Textbox(label="General Prompt")
                            alpha_blend = gr.Slider(label="Alpha Blend", minimum=0.0, maximum=1.0, value=0.2, step=0.01, interactive=True)

                            for n in range(MAX_COLORS):
                                with gr.Row(visible=False) as color_row[n]:

                                    with gr.Box(elem_id="color-bg"):
                                        colors.append(gr.HTML(
                                            '<div class="color-bg-item" style="background-color: black"></div>'))
                                    with gr.Column():
                                        with gr.Row():
                                            prompts.append(gr.Textbox(label="Prompt for this mask"))

                                        with gr.Row():
                                            weight_slider = gr.Slider(label=f"Area {n+1} Weight", minimum=0.0, maximum=1.0,
                                                                    value=1.0, step=0.01, interactive=True, elem_id=f"weight_{n+1}_slider")
                                            cur_weight_sliders.append(weight_slider)

                            button_update = gr.Button("Prompt Info Update", elem_id="update_button", interactive=True)
                            final_prompt = gr.Textbox(label="Final Prompt", interactive=False)

                        button_run.click(process_sketch, inputs=[canvas_image, binary_matrixes],
                                         outputs=[post_sketch, binary_matrixes, alpha_mask_row, alpha_color, *color_row, *colors],
                                         queue=False)

                        button_update.click(fn=update_mask_filters, inputs=[alpha_blend, general_prompt, *cur_weight_sliders, *prompts], outputs=[final_prompt, alpha_mask_row, alpha_color, *colors])

                        def paste_prompt(*input_prompts):
                            final_prompts = input_prompts[:len(self.area_colors)]
                            final_prompt_str = '\nAND '.join(final_prompts)
                            return final_prompt_str
                        source_prompts = [general_prompt, *prompts]
                        button_update.click(fn=paste_prompt, inputs=source_prompts,
                                            outputs=self.target_paste_prompt)



                        with gr.Column():
                            canvas_width = gr.Slider(label="Canvas Width", minimum=256, maximum=1024, value=512, step=64)
                            canvas_height = gr.Slider(label="Canvas Height", minimum=256, maximum=1024, value=512, step=64)


                            canvas_swap_res = ToolButton(value=switch_values_symbol)
                            canvas_swap_res.click(lambda w, h: (h, w), inputs=[canvas_width, canvas_height],
                                                  outputs=[canvas_width, canvas_height])
                        create_button = gr.Button(value="Create blank canvas")
                        create_button.click(fn=create_canvas, inputs=[canvas_height, canvas_width], outputs=[canvas_image])

                    with gr.TabItem("Rectangular", elem_id="tab_twoshot_rect") as twoshot_tab_rect:
                        with gr.Row():
                            divisions = gr.Textbox(label="Divisions", elem_id=f"cd_{id_part}_divisions", value="1:1,1:2,1:2")
                            positions = gr.Textbox(label="Positions", elem_id=f"cd_{id_part}_positions", value="0:0,0:0,0:1")
                        with gr.Row():
                            weights = gr.Textbox(label="Weights", elem_id=f"cd_{id_part}_weights", value="0.2,0.8,0.8")
                            end_at_step = gr.Slider(minimum=0, maximum=150, step=1, label="end at this step", elem_id=f"cd_{id_part}_end_at_this_step", value=150)

                        visualize_button = gr.Button(value="Visualize")
                        visual_regions = gr.Gallery(label="Regions").style(grid=(4, 4, 4, 8), height="auto")

                        visualize_button.click(fn=self.do_visualize, inputs=[divisions, positions, weights], outputs=[visual_regions])

                        extra_generation_params = gr.Textbox(label="Extra generation params")
                        apply_button = gr.Button(value="Apply")

                        apply_button.click(fn=self.do_apply, inputs=[extra_generation_params], outputs=[divisions, positions, weights, end_at_step])

                    def select_twosoht_tab(tab_id):
                        self.selected_twoshot_tab = tab_id
                    for i, elem in enumerate(
                            [twoshot_tab_mask, twoshot_tab_rect]):
                        elem.select(
                            fn=lambda tab=i: select_twosoht_tab(tab),
                            inputs=[],
                            outputs=[],
                    )

        self.ui_root = group_two_shot_root

        self.infotext_fields = [
            (extra_generation_params, "Latent Couple")
        ]
        process_script_params.append(enabled)
        process_script_params.append(divisions)
        process_script_params.append(positions)
        process_script_params.append(weights)
        process_script_params.append(end_at_step)
        process_script_params.append(alpha_blend)
        process_script_params.extend(cur_weight_sliders)
        return process_script_params

    def denoised_callback(self, params: CFGDenoisedParams):

        if self.enabled and params.sampling_step < self.end_at_step:

            x = params.x
            # x.shape = [batch_size, C, H // 8, W // 8]

            num_batches = self.num_batches
            num_prompts = x.shape[0] // num_batches
            # ex. num_batches = 3
            # ex. num_prompts = 3 (tensor) + 1 (uncond)

            if self.debug:
                print(f"### Latent couple ###")
                print(f"denoised_callback x.shape={x.shape} num_batches={num_batches} num_prompts={num_prompts}")

            filters = [
                f.create_tensor(x.shape[1], x.shape[2], x.shape[3]) for f in self.filters
            ]
            neg_filters = [1.0 - f for f in filters]

            """
            batch #1
              subprompt #1
              subprompt #2
              subprompt #3
            batch #2
              subprompt #1
              subprompt #2
              subprompt #3
            uncond
              batch #1
              batch #2
            """

            tensor_off = 0
            uncond_off = num_batches * num_prompts - num_batches
            for b in range(num_batches):
                uncond = x[uncond_off, :, :, :]

                for p in range(num_prompts - 1):
                    if self.debug:
                        print(f"b={b} p={p}")
                    if p < len(filters):
                        tensor = x[tensor_off, :, :, :]
                        x[tensor_off, :, :, :] = tensor * filters[p] + uncond * neg_filters[p]

                    tensor_off += 1

                uncond_off += 1

    def process(self, p: StableDiffusionProcessing, *args, **kwargs):

        enabled, raw_divisions, raw_positions, raw_weights, raw_end_at_step, alpha_blend, *cur_weight_sliders = args

        self.enabled = enabled

        if not self.enabled:
            return

        self.num_batches = p.batch_size

        if self.selected_twoshot_tab == 0:
            pass
        elif self.selected_twoshot_tab == 1:
            self.filters = self.create_rect_filters_from_ui_params(raw_divisions, raw_positions, raw_weights)
        else:
            raise ValueError(f"Unknown filter mode")

        self.end_at_step = raw_end_at_step

        # TODO: handle different cases for generation info: 'mask' and 'rect'
        # if self.end_at_step != 0:
        #     p.extra_generation_params["Latent Couple"] = f"divisions={raw_divisions} positions={raw_positions} weights={raw_weights} end at step={raw_end_at_step}"


        if self.debug:
            print(f"### Latent couple ###")
            print(f"process num_batches={self.num_batches} end_at_step={self.end_at_step}")

        if not hasattr(self, 'callbacks_added'):
            on_cfg_denoised(self.denoised_callback)
            self.callbacks_added = True

        return

    def postprocess(self, *args):
        return


script_callbacks.on_after_component(prompt_textbox_tracker.on_after_component_callback)