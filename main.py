import json
import platform
from pprint import pprint
import shutil
from pydub import AudioSegment
from pydub.silence import detect_silence
from typing import Literal
import os
import whisper
import subprocess

from custom_modules.audio_separator import Separator
from custom_modules.speaker.lib_speak import Speaker


def path_to(*els: str):
    return os.path.join(os.path.dirname(__file__), *els)


ffmpeg_path = path_to(
    'ffmpeg.exe') if platform.system() == 'Windows' else 'ffmpeg'

print(ffmpeg_path)



def get_vocal_and_instrumental_from_video(fname: str, input_path: str, output_path: str, vocal_path: str, instrumental_path: str):
    subprocess.call([ffmpeg_path, '-i', input_path, '-vn', '-acodec',
                    'pcm_s16le', '-ar', '44100', '-ac', '1', os.path.join(output_path, f'{fname}.wav'), '-y'])

    print('Separating audio...')
    # Initialize the Separator with the audio file and model name
    separator = Separator(os.path.join(output_path, f'{fname}.wav'), model_name='UVR_MDXNET_KARA_2', use_cuda=True,
                          model_file_dir=path_to('models', 'asm'), primary_stem_path=instrumental_path, secondary_stem_path=vocal_path)

    # Perform the separation
    primary_stem_path, secondary_stem_path = separator.separate()

    print(f'Background sounds saved at {primary_stem_path}')
    print(f'Speech sound saved at {secondary_stem_path}')

    print('Audio separated')

    print('Converting to mono...')
    sound = AudioSegment.from_wav(secondary_stem_path)
    sound = sound.set_channels(1)
    sound.export(secondary_stem_path, format="wav")
    print('Converted to mono')
    return primary_stem_path, secondary_stem_path
    # return instrumental_path, vocal_path


def translate_video(*, input_path: str, output_path: str, language: Literal["en"]):
    fname = '.'.join(os.path.basename(input_path).split('.')[:-1])
    instrumental, vocal = get_vocal_and_instrumental_from_video(
        fname, input_path, output_path, os.path.join(output_path, 'vocal.wav'), os.path.join(output_path, 'instrumental.wav'))

    model = whisper.load_model(
        "medium", device='cuda', download_root=path_to('models', 'openai'))

    # language is initial language in video
    result = model.transcribe(vocal, task='translate', language="Russian")

    # translating to another languages
    if not language == 'en':
        ...

    speaker = Speaker(model_id="v3_en", language="en",
                      speaker="en_1", device="cuda")  # type: ignore

    if os.path.exists(os.path.join(output_path, 'out_mp3')):
        shutil.rmtree(os.path.join(output_path, 'out_mp3'))

    os.makedirs(os.path.join(output_path, 'out_mp3'), exist_ok=True)


    speaker.to_mp3(text=result['text'], name_text=fname,  # type: ignore
                   sample_rate=48000, audio_dir=os.path.join(output_path, 'out_mp3'), speed=1.0)


    audio_file = os.path.join(output_path, 'out_mp3',
                              [i for i in os.listdir(os.path.join(output_path, 'out_mp3')) if i !=
                               'cache' and i.endswith('.mp3')][0])

    sound2 = AudioSegment.from_file(audio_file)
    original_vocal = AudioSegment.from_file(vocal)

    offset_by_silent_fragments = 0

    non_speech_fragments = detect_silence(
        original_vocal, min_silence_len=500, silence_thresh=-50)
    # Print the positions and durations of the non-speech fragments
    for fragment in non_speech_fragments:
        silent = AudioSegment.silent(duration=abs(
            fragment[1]-fragment[0]), frame_rate=sound2.frame_rate)
        s1 = sound2[:fragment[0]]
        s2 = sound2[fragment[0]:]
        sound2 = s1 + original_vocal[fragment[0]:fragment[1]] + s2
        offset_by_silent_fragments += abs(
            fragment[1]-fragment[0])

    print(offset_by_silent_fragments, abs(
        sound2.duration_seconds - original_vocal.duration_seconds))

    sound1 = AudioSegment.from_file(instrumental)
    combined = sound1.overlay(sound2)

    combined.export(os.path.join(output_path, fname+'.new.mp3'), format='mp3')

    subprocess.call(['ffmpeg', '-i', input_path, '-i', os.path.join(output_path, fname+'.new.mp3'), '-c:v', 'copy',
                    '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', os.path.join(output_path, fname+'.translated.mp4')])

p = path_to('18.mp4')
translate_video(input_path=p, output_path=path_to('output'), language='en')
