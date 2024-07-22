import base64
import datetime
import json
import io
import requests

import modules.scripts as scripts
import gradio as gr

from modules.processing import process_images, program_version
from modules.shared import opts, state, OptionInfo
from modules import script_callbacks

task_api = '/api/v1/task'
sub_task_api = '/api/v1/sub_task'

t_txt2img = "txt2img"
t_img2img = "img2img"


class Gen2Gallery(scripts.Script):
    def title(self):
        return "Gen2Gallery"

    # 显示的条件
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # ui显示
    def ui(self, is_img2img):
        return []

    # 运行的时候嵌入运行
    def run(self, p, *args):
        proc = process_images(p)
        return proc

    def before_process(self, p, *args):
        gen2_service_option = {
            "enable": self.has_enable_sync(),
            "server_url": opts.ai_gallery_service_url,
            "username": opts.ai_gallery_username,
            "password": opts.ai_gallery_password,
            "headers": {
                "X-Ag-H-U": json.dumps({
                    "username": opts.ai_gallery_username,
                    "password": opts.ai_gallery_password
                })
            }
        }
        setattr(p, "gen2_service_option", gen2_service_option)

    def process(self, p, *args):
        category = self.get_generate_category()
        ref_images = []
        if self.is_img2img:
            for index, img in enumerate(p.init_images):
                base64_str = to_img_base64(img)
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")  # 格式化时间戳
                filename = f"{timestamp}_ref_image_{index}"
                ref_images.append({
                    "base64": base64_str,
                    "filename": filename
                })
        if p.gen2_service_option["enable"]:
            body = {
                "category": category,
                "ref_images": ref_images,
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                "width": p.width,
                "height": p.height,
                "seed": f"{p.seed}",
                "sampler_name": p.sampler_name,
                "cfg_scale": p.cfg_scale,
                "steps": p.steps,
                "batch_size": p.batch_size,
                "total": len(p.all_prompts),
                "sd_model_name": p.sd_model_name,
                "sd_model_hash": p.sd_model_hash,
                "sd_vae_name": p.sd_vae_name,
                "sd_vae_hash": p.sd_vae_hash,
                "job_timestamp": state.job_timestamp,
                "version": program_version()
            }
            try:
                response = requests.post(p.gen2_service_option["server_url"] + task_api,
                                         headers=p.gen2_service_option["headers"],
                                         json=body)
                response.raise_for_status()
                setattr(p, "gen2_server_task_id", json.loads(response.text)["task_id"])
            except requests.RequestException as e:
                print(f"process function request failed, {e}, error: {response.text}")

    def has_enable_sync(self) -> bool:
        return (self.is_img2img or self.is_txt2img) and opts.ai_gallery_enable == 'open'

    def postprocess_image_after_composite(self, p, pp, *args):
        setattr(pp.image, "gen2_is_grid_image", False)

    def postprocess(self, p, processed, *args):
        if not hasattr(p, "gen2_server_task_id"):
            return
        body = {}
        # 保存网格的图片
        if len(processed.images) > 0 and hasattr(processed.images[0], "already_saved_as"):
            body = {
                "base64": to_img_base64(processed.images[0]),
                "filename": processed.images[0].already_saved_as,
                "extra": processed.js()
            }
        else:
            body = {"extra": processed.js()}

        url = f"{p.gen2_service_option['server_url']}{task_api}/{p.gen2_server_task_id}"
        try:
            response = requests.put(url, headers=p.gen2_service_option["headers"], json=body)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"postprocess function request failed, {e}, error: {response.text}")

    def get_generate_category(self) -> str:
        if self.is_img2img:
            return t_img2img
        else:
            return t_txt2img


def to_img_base64(img) -> str:
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    img_bytes = img_byte_arr.read()
    img_base64 = base64.b64encode(img_bytes).decode('ascii')
    return img_base64


def on_save_image(params):
    p = params.p
    img = params.image
    if not hasattr(p, "gen2_server_task_id"):
        return
    # 保存生成的图片
    if hasattr(img, "gen2_is_grid_image"):
        body = {
            "task_id": p.gen2_server_task_id,
            "base64": to_img_base64(img),
            "filename": params.filename
        }
        try:
            response = requests.post(p.gen2_service_option["server_url"] + sub_task_api,
                                     headers=p.gen2_service_option["headers"], json=body)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"on_save_image function request failed, {e}, error: {response.text}")


def on_ui_settings():
    section = ('ai_gallery', "AI 图库")

    opts.add_option(
        "ai_gallery_enable",
        OptionInfo(
            "close",  # 默认值为关闭
            "是否开启上传功能",
            gr.Radio,
            {"choices": ["open", "close"]},
            section=section
        )
    )
    opts.add_option(
        "ai_gallery_service_url",
        OptionInfo(
            "",
            "服务器地址",
            gr.Textbox,
            {"interactive": True},
            section=section
        ),
    )
    opts.add_option(
        "ai_gallery_username",
        OptionInfo(
            "",
            "用户名",
            gr.Textbox,
            {"interactive": True},
            section=section
        )
    )
    opts.add_option(
        "ai_gallery_password",
        OptionInfo(
            "",
            "密码",
            gr.Textbox,
            {"interactive": True},
            section=section
        )
    )


script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_image_saved(on_save_image)
