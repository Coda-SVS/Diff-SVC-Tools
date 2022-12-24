import os
from typing import List

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import subprocess
import glob
from infer import *
import logging
from infer_tools.infer_tool import *

from tqdm import tqdm

print = tqdm.write

logging.getLogger("numba").setLevel(logging.WARNING)

__LOADED_MODEL = {"svc_model": None, "project_name": None, "model_path": None, "check_point_num": None}


def get_svc_model(project_name: str, hubert_gpu: bool = True):
    if __LOADED_MODEL["project_name"] == project_name:
        return __LOADED_MODEL["svc_model"], __LOADED_MODEL["model_path"], __LOADED_MODEL["check_point_num"]

    checkpoints = glob.glob(f"checkpoints/{project_name}/*.ckpt")

    ckpt_pathdict = {}
    for ckpt_path in checkpoints:
        ckpt_num = int(os.path.splitext(os.path.basename(ckpt_path))[0].split("_")[-1])
        ckpt_pathdict[ckpt_num] = ckpt_path

    key_list = list(ckpt_pathdict.keys())
    key_list.sort()

    check_point_num = key_list[-1]
    model_path = ckpt_pathdict[check_point_num]

    print(f"{project_name=}")
    print(f"{check_point_num=}")

    config_path = os.path.join(os.path.dirname(model_path), "config.yaml")

    svc_model = Svc(project_name, config_path, hubert_gpu, model_path)
    print("model loaded")

    __LOADED_MODEL["project_name"] = project_name
    __LOADED_MODEL["svc_model"] = svc_model
    __LOADED_MODEL["model_path"] = model_path
    __LOADED_MODEL["check_point_num"] = check_point_num

    return __LOADED_MODEL["svc_model"], __LOADED_MODEL["model_path"], __LOADED_MODEL["check_point_num"]


class SynthesisRequest:
    def __init__(self, input_path, key: int = 0, pndm_speedup: int = 5, add_noise_step: int = 0, use_gt_mel: bool = False, out_dir: str = "results") -> None:
        self.input_path = input_path
        self.key = key
        self.pndm_speedup = pndm_speedup
        self.add_noise_step = add_noise_step
        self.out_dir = out_dir
        self.use_gt_mel = use_gt_mel

    def synthesis(self, project_name: str, hubert_gpu: bool = True, use_deepfilter: bool = False):
        svc_model, model_path, check_point_num = get_svc_model(project_name=project_name, hubert_gpu=hubert_gpu)

        os.makedirs(self.out_dir, exist_ok=True)

        wav_gen = os.path.join(
            self.out_dir,
            f"{project_name}_"
            f"{check_point_num}_"
            f"{os.path.splitext(os.path.basename(self.input_path))[0]}_"
            f'{f"key_{self.key}_" if not self.use_gt_mel else ""}'
            f'{f"noise_{self.add_noise_step}_" if self.use_gt_mel else ""}'
            f'{"use_gt_mel_" if self.use_gt_mel else ""}'
            "output.flac",
        )

        f0_tst, f0_pred, audio = run_clip(
            svc_model,
            file_path=self.input_path,
            key=self.key,
            acc=self.pndm_speedup,
            use_crepe=True,
            use_pe=False,
            thre=0.05,
            use_gt_mel=self.use_gt_mel,
            add_noise_step=self.add_noise_step,
            project_name=project_name,
            out_path=wav_gen,
        )

        if use_deepfilter:
            filename = os.path.splitext(os.path.basename(wav_gen))[0] + "_deepfilter"

            subprocess.run(f'ffmpeg -i "{wav_gen}" -ar 48000 "{filename + ".wav"}" -y', shell=True)
            subprocess.run(f'deepfilter.exe "{filename + ".wav"}" -o "{self.out_dir}"', shell=True)
            subprocess.run(f'ffmpeg -i "{os.path.join(self.out_dir, filename + ".wav")}" "{os.path.join(self.out_dir, filename + ".flac")}" -y', shell=True)

            os.remove(os.path.join(self.out_dir, filename + ".wav"))
            os.remove(filename + ".wav")


def main():
    # 여러 모델을 사용할 경우, 한 번에 모든 모델을 사용할 수 있음
    model_project_names: List[str] = [
        "model_name1",
        # "model_name2",
    ]

    # raw 폴더내의 wav 파일을 모두 가져옴
    raw_audio_filepaths = glob.glob("raw/*.wav", recursive=True)
    raw_audio_filepaths += glob.glob("raw/**/*.wav", recursive=True)
    raw_audio_filepaths = list(set(raw_audio_filepaths))  # 중복 제거
    raw_audio_filepaths.sort()  # 정렬

    raw_audio_files: List[SynthesisRequest] = []

    global_key = 0  # 전체적인 키 변경이 필요할 경우

    # 단일 파일당 옵션을 생성
    for filepath in raw_audio_filepaths:
        raw_audio_files.append(SynthesisRequest(filepath))
        # raw_audio_files.append(SynthesisRequest(filepath, pndm_speedup=1, use_gt_mel=True, add_noise_step=1000))  # 고퀄리티 모드 (느림) (키 변환 지원 안됨)

    # 직접 수동으로 wav 파일을 지정할 수 있음
    # raw_audio_files: List[SynthesisRequest] = [
    #     # 보컬 음원 파일 이름을 꼭 바꿔주세요
    #     SynthesisRequest("raw/vocal_file.wav", key=2),  # 키 변경이 필요할 경우
    #     SynthesisRequest("raw/vocal_file.wav", pndm_speedup=1, use_gt_mel=True, add_noise_step=1000),  # 고퀄리티 모드 (느림)
    # ]

    # infer 시작
    tqdm.write(f"{global_key=}")
    for synthesis_req in (synthesis_req_tqdm := tqdm(raw_audio_files, leave=False)):
        synthesis_req_tqdm.set_description(f"Processing... [{synthesis_req.input_path}]")
        for model_project_name in (model_project_name_tqdm := tqdm(model_project_names, leave=False)):
            model_project_name_tqdm.set_description(f"Processing... [{model_project_name}]")
            synthesis_req.key += global_key
            synthesis_req.synthesis(project_name=model_project_name)


if __name__ == "__main__":
    main()
