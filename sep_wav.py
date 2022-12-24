import os
from typing import List
from tqdm import tqdm
from glob import glob
import subprocess

import numpy as np
import librosa
import soundfile
from pydub import AudioSegment, effects

temp_log_path = "temp_ffmpeg_log.txt"  # ffmpeg의 무음 감지 로그의 임시 저장 위치


def audio_norm(input_filepath: str, output_filepath: str):
    """오디오 파일에 노멀라이징 효과를 적용합니다.

    Args:
        input_filepath (str): 입력 파일의 경로
        output_filepath (str): 효과가 적용된 오디오 파일의 출력 경로
    """

    ext = os.path.splitext(input_filepath)[1][1:]

    assert ext in ["wav", "flac"], "지원하지 않는 포멧입니다."

    rawsound = AudioSegment.from_file(input_filepath, format=ext)
    normalizedsound = effects.normalize(rawsound)
    normalizedsound.export(output_filepath, format="flac")


def get_ffmpeg_args(filepath: str) -> str:
    """ffmpeg의 명령줄을 생성합니다.

    Args:
        filepath (str): 파일 경로

    Returns:
        str: ffmpeg 인자값이 포함된 명령줄
    """

    global temp_log_path

    return f'ffmpeg -i "{filepath}" -af "silencedetect=n=-50dB:d=1.5,ametadata=print:file={temp_log_path}" -f null -'


def get_audiofiles(path: str) -> List[str]:
    """해당 폴더 내부의 모든 오디오 파일을 가져옵니다. (flac, wav만 지원)

    Args:
        path (str): 폴더 위치

    Returns:
        List[str]: 오디오 파일의 경로
    """

    filepaths = glob(os.path.join(path, "**", "*.flac"), recursive=True)
    filepaths += glob(os.path.join(path, "*.flac"), recursive=True)
    filepaths += glob(os.path.join(path, "**", "*.wav"), recursive=True)
    filepaths += glob(os.path.join(path, "*.wav"), recursive=True)
    filepaths = list(set(filepaths))
    filepaths.sort()

    return filepaths


def main(input_dir: str, output_dir: str, split_sil: bool = False, use_norm: bool = True) -> None:
    """메인 로직

    Args:
        input_dir (str): 오디오 파일의 원본 위치 (폴더)
        output_dir (str): 처리가 완료된 오디오 파일의 출력 위치 (최종본은 final 폴더에 저장됨)
        split_sil (bool, optional): 오디오 파일에서 부분적인 무음을 잘라냅니다. Defaults to False.
        use_norm (bool, optional): 오디오 노멀라이징을 적용합니다. Defaults to True.
    """

    filepaths = get_audiofiles(input_dir)

    output_final_dir = os.path.join(output_dir, "final")
    os.makedirs(output_final_dir, exist_ok=True)

    if use_norm:
        output_norm_dir = os.path.join(output_dir, "norm")
        os.makedirs(output_norm_dir, exist_ok=True)

        for filepath in tqdm(filepaths, desc="노멀라이징 작업 중..."):
            filename = os.path.splitext(os.path.basename(filepath))[0]
            out_filepath = os.path.join(output_norm_dir, filename) + ".wav"
            audio_norm(filepath, out_filepath)

        filepaths = get_audiofiles(output_norm_dir)

    for filepath in tqdm(filepaths, desc="음원 자르는 중..."):
        duration = librosa.get_duration(filename=filepath)
        max_last_seg_duration = 0
        sep_duration_final = 15
        sep_duration = 15

        while sep_duration > 4:
            last_seg_duration = duration % sep_duration
            if max_last_seg_duration < last_seg_duration:
                max_last_seg_duration = last_seg_duration
                sep_duration_final = sep_duration
            sep_duration -= 1

        filename = os.path.splitext(os.path.basename(filepath))[0]
        out_filepath = os.path.join(output_final_dir, f"{filename}-%03d.wav")
        subprocess.run(f'ffmpeg -i "{filepath}" -f segment -segment_time {sep_duration_final} "{out_filepath}" -y', capture_output=True, shell=True)

    filepaths = get_audiofiles(output_final_dir)

    for filepath in tqdm(filepaths, desc="무음 제거 중..."):
        if os.path.exists(temp_log_path):
            os.remove(temp_log_path)

        ffmpeg_arg = get_ffmpeg_args(filepath)
        subprocess.run(ffmpeg_arg, capture_output=True, shell=True)

        start = None
        end = None

        with open(temp_log_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip()
                if "lavfi.silence_start" in line:
                    start = float(line.split("=")[1])
                if "lavfi.silence_end" in line:
                    end = float(line.split("=")[1])

        if start != None:
            if start == 0 and end == None:
                os.remove(filepath)
            elif split_sil:
                if end == None:
                    end = len(y)
                else:
                    end = int(end)

                y, sr = librosa.load(filepath, sr=None)
                y = np.concatenate((y[: round(sr * start)], y[round(sr * end) :]), axis=None)
                soundfile.write(filepath, y, samplerate=sr)

    if os.path.exists(temp_log_path):
        os.remove(temp_log_path)


if __name__ == "__main__":
    input_dir = "preprocess"
    output_dir = "preprocess_out"
    split_sil = False
    use_norm = True

    main(
        input_dir=input_dir,
        output_dir=output_dir,
        split_sil=split_sil,
        use_norm=use_norm,
    )
