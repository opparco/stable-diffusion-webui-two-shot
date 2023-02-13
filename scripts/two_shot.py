from typing import List

import torch

from modules import devices

import modules.scripts as scripts
import gradio as gr

# todo:
from modules.script_callbacks import CFGDenoisedParams, on_cfg_denoised

# from modules import processing
# from torchvision import transforms

from modules.processing import StableDiffusionProcessing


class Filter:
    def __init__(self, division: (int, int), position: (int, int), weight: float):
        self.division = division
        self.position = position
        self.weight = weight

    def create_tensor(
        self, num_channels: int, height_b: int, width_b: int
    ) -> torch.Tensor:

        x = torch.zeros(num_channels, height_b, width_b).to(devices.device)

        dy, dx = self.division
        py, px = self.position

        division_height = height_b // dy
        division_width = width_b // dx
        y1 = division_height * py
        x1 = division_width * px

        x[:, y1 : y1 + division_height, x1 : x1 + division_width] = self.weight

        return x


class Script(scripts.Script):
    def __init__(self):
        self.num_batches: int = 0
        self.end_at_step: int = 20
        self.filters: List[Filter] = []
        self.debug: bool = False
        self.enabled: bool = False

    def title(self):
        return "Latent Couple extension"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def create_filters_from_ui_params(self, raw_divisions, raw_positions, raw_weights):

        divisions = []
        for division in raw_divisions.split(","):
            y, x = division.split(":")
            divisions.append((int(y), int(x)))

        positions = []
        for position in raw_positions.split(","):
            y, x = position.split(":")
            positions.append((int(y), int(x)))

        weights = []
        for w in raw_weights.split(","):
            weights.append(float(w))

        # todo: assert len

        return [
            Filter(division, position, weight)
            for division, position, weight in zip(divisions, positions, weights)
        ]

    def do_visualize(self, raw_divisions, raw_positions, raw_weights):

        self.filters = self.create_filters_from_ui_params(
            raw_divisions, raw_positions, raw_weights
        )

        return [
            f.create_tensor(1, 128, 128).squeeze(dim=0).cpu().numpy()
            for f in self.filters
        ]

    def ui(self, is_img2img):
        id_part = "img2img" if is_img2img else "txt2img"

        with gr.Group():
            with gr.Accordion("Latent Couple", open=False):
                enabled = gr.Checkbox(value=False, label="Enabled")
                with gr.Row():
                    divisions = gr.Textbox(
                        label="Divisions",
                        elem_id=f"cd_{id_part}_divisions",
                        value="1:1,1:2,1:2",
                    )
                    positions = gr.Textbox(
                        label="Positions",
                        elem_id=f"cd_{id_part}_positions",
                        value="0:0,0:0,0:1",
                    )
                with gr.Row():
                    weights = gr.Textbox(
                        label="Weights",
                        elem_id=f"cd_{id_part}_weights",
                        value="0.2,0.8,0.8",
                    )
                    end_at_step = gr.Slider(
                        minimum=0,
                        maximum=150,
                        step=1,
                        label="end at this step",
                        elem_id=f"cd_{id_part}_end_at_this_step",
                        value=20,
                    )

                visualize_button = gr.Button(value="Visualize")
                visual_regions = gr.Gallery(label="Regions").style(
                    grid=(4, 4, 4, 8), height="auto"
                )

                visualize_button.click(
                    fn=self.do_visualize,
                    inputs=[divisions, positions, weights],
                    outputs=[visual_regions],
                )

        return enabled, divisions, positions, weights, end_at_step

    def denoised_callback(self, params: CFGDenoisedParams):
        if not self.enabled:
            return

        if params.sampling_step < self.end_at_step:

            x = params.x
            # x.shape = [batch_size, C, H // 8, W // 8]

            num_batches = self.num_batches
            num_prompts = x.shape[0] // num_batches
            # ex. num_batches = 3
            # ex. num_prompts = 3 (tensor) + 1 (uncond)

            if self.debug:
                print(f"### Latent couple ###")
                print(
                    f"denoised_callback x.shape={x.shape} num_batches={num_batches} num_prompts={num_prompts}"
                )

            filters = [
                f.create_tensor(x.shape[1], x.shape[2], x.shape[3])
                for f in self.filters
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
                        x[tensor_off, :, :, :] = (
                            tensor * filters[p] + uncond * neg_filters[p]
                        )

                    tensor_off += 1

                uncond_off += 1

    def process(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        raw_divisions: str,
        raw_positions: str,
        raw_weights: str,
        raw_end_at_step: int,
    ):
        self.enabled = enabled
        if not self.enabled:
            return

        self.num_batches = p.batch_size

        self.filters = self.create_filters_from_ui_params(
            raw_divisions, raw_positions, raw_weights
        )

        self.end_at_step = raw_end_at_step

        #

        if self.end_at_step != 0:
            p.extra_generation_params[
                "Latent Couple"
            ] = f"divisions={raw_divisions} positions={raw_positions} weights={raw_weights} end at step={raw_end_at_step}"
            # save params into the output file as PNG textual data.

        if self.debug:
            print(f"### Latent couple ###")
            print(
                f"process num_batches={self.num_batches} end_at_step={self.end_at_step}"
            )

        if not hasattr(self, "callbacks_added"):
            on_cfg_denoised(self.denoised_callback)
            self.callbacks_added = True

        return

    def postprocess(self, *args):
        return
