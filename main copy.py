import platform
from pprint import pprint
from pydub import AudioSegment
from typing import Literal
import os
import whisper
import subprocess

from voice_cloning.generation import speech_generator, save_sound

from audio_separator import Separator
from speakerpy.lib_speak import Speaker
from typing import Optional, Tuple

ffmpeg_path = os.path.join(os.getcwd(), "file.exe") if (
    platform.system() == 'Windows') else os.path.join(os.getcwd(), "ffmpeg")

# Функция для конвертации аудио в моно


def convert_to_mono(wav_file_path: str) -> None:
    print('Конвертация в моно...')
    sound = AudioSegment.from_wav(wav_file_path)
    sound = sound.set_channels(1)
    sound.export(wav_file_path, format="wav")
    print('Конвертация в моно завершена')

# Функция для разделения аудио из видео


def separate_audio_from_video(video_file_path: str, vocal_path: str, instrumental_path: str) -> Optional[tuple[str, str]]:
    try:
        # Используем ffmpeg для извлечения аудио из видео
        subprocess.call(['ffmpeg', '-i', f'{video_file_path}', '-vn', '-acodec',
                         'pcm_s16le', '-ar', '44100', '-ac', '2', f'{video_file_path}.wav'])

        print('Разделение аудио...')
        # Инициализируем Separator с аудиофайлом и указываем пути для сохранения вокала и инструментала
        separator = Separator(f'{video_file_path}.wav', model_name='UVR_MDXNET_KARA_2', use_cuda=True,
                              model_file_dir='asm/', primary_stem_path=instrumental_path, secondary_stem_path=vocal_path)

        # Выполняем разделение
        primary_stem_path, secondary_stem_path = separator.separate()

        print(f'Фоновые звуки сохранены в {primary_stem_path}')
        print(f'Речевой звук сохранен в {secondary_stem_path}')
        print('Аудио разделено')

        # Конвертируем вокальную часть в моно
        convert_to_mono(secondary_stem_path)

        return primary_stem_path, secondary_stem_path

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        return None


def translate_video(*, input_path: str, output_path: str, language: Literal["en"]):
    fname = '.'.join(os.path.basename(input_path).split('.')[:-1])
    model = whisper.load_model("medium", device='cuda', download_root="openai")
    try:
        result_separating = separate_audio_from_video(
            fname, os.path.join(output_path, 'vocal.wav'), os.path.join(output_path, 'instrumental.wav'))
        if result_separating is not None:
            instrumental, vocal = result_separating
            # language is initial language in video
            result = model.transcribe(
                vocal, task='translate', language="Russian")
            pprint(result)
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")

    # get all fragments without speech

    #

    # translating to another languages
    if not language == 'en':
        ...

    # speaker = Speaker(model_id="v3_en", language="en",
    #                   speaker="en_100", device="cuda")  # type: ignore
    # speaker.to_mp3(text=result['text'], name_text=fname,  # type: ignore
    #                sample_rate=48000, audio_dir=os.path.join(output_path, 'out_mp3'), speed=1.0)

    generated_wav = speech_generator(
        voice_type="western",  # supports "indian" & "western"
        sound_path=vocal,
        speech_text=result['text']
    )

    save_sound(generated_wav, filename=os.path.join(output_path, fname+'.synthesized.wav'),  # type: ignore
               noise_reduction=True)  # enable noise reduction

    # audio_file = os.path.join(output_path, 'out_mp3' +
    #                           [i for i in os.listdir(os.path.join(output_path, 'out_mp3')) if i !=
    #                            'cache' and i.endswith('.mp3')][0])
    # print(audio_file)

    # sound1 = AudioSegment.from_file(instrumental)
    # sound2 = AudioSegment.from_file(audio_file)
    # sound2 = sound2 - 15
    # sound1 = sound1 - 15
    # combined = sound1.overlay(sound2)
    # combined.export(fname+'.new.mp3', format='mp3')

    # subprocess.call(['ffmpeg', '-i', input_path, '-i', fname+'.new.mp3', '-c:v', 'copy',
    #                 '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', os.path.join(output_path, fname+'.translated.mp4')])


p = os.getcwd() + '58.mp4'
translate_video(input_path='58.mp4', output_path='./output/', language='en')
