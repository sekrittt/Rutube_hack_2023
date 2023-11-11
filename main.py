# Импорт библиотек
import logging
import shutil
from tkinter import NO
from pydub import AudioSegment
from pydub.silence import detect_silence
from typing import Literal, Optional, Union
import os
import whisper
import subprocess
from colorama import Fore

from argparse import ArgumentParser

# Откорректированные модули
from custom_modules.audio_separator import Separator
from custom_modules.speaker.lib_speak import Speaker

from argostranslate import package as arg_package
from argostranslate import translate as arg_translate

model_sr: whisper.Whisper
device: Literal['cpu', 'cuda'] = 'cpu'
Langs = Literal['en', 'fr']


def path_to(*els: str, base_path: Optional[str] = None) -> str:
    """
    Функция для создания пути и директорий.

    Parameters:
        *els: str - Элементы для добавления к пути.
        base_path: str - Базовый путь. Если не указан, используется текущая директория.

    Returns:
        str: Полный путь к созданным директориям.
    """
    base_path = base_path or os.path.dirname(__file__)
    full_path = os.path.join(base_path, *els)

    os.makedirs(full_path, exist_ok=True)

    return full_path


def get_audio(*, input_path: str, output_dir: str) -> Optional[str]:
    """
    Извлекает аудио из видео-файла с использованием ffmpeg.

    Parameters:
        input_path (str): Путь к исходному видео-файлу.

    Returns:
        str: Полный путь к аудио-файлу
    """
    print(Fore.BLUE+"Извлечение аудио из видео..."+Fore.WHITE)
    try:
        output_path = os.path.join(output_dir, '.'.join(
            input_path.split('.')[:-1])+'.wav')
        subprocess.run(['ffmpeg', '-i', input_path, '-vn', '-acodec',
                        'pcm_s16le', '-ar', '44100', '-ac', '1', output_path, '-y', '-loglevel', 'quiet'], check=True)
        print(Fore.BLUE+f"Aудио успешно извлечено: {output_path}"+Fore.WHITE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(Fore.RED+f"Ошибка при извлечении аудио: {e}")
        return None


def split_audio(*, input_path: str, output_path: str) -> Optional[tuple[str, str]]:
    global device
    """
    Разделяет аудио на фоновую и основную части с использованием Splitter.

    Parameters:
        input_path (str): Путь к исходному аудио-файлу.
        device (str): Устройство для выполнения разделения ('cpu' или 'cuda').

    Returns:
        tuple[str, str]: Пути к файлам с фоновой и основной частями.
    """
    print(Fore.BLUE+"Разделение аудио..."+Fore.WHITE)
    try:
        separator = Separator(input_path, model_name='UVR_MDXNET_KARA_2', use_cuda=True if device == 'cuda' else False,
                              model_file_dir=path_to('models', 'asm'), output_dir=output_path, log_level=logging.FATAL)

        # Perform the separation
        primary_stem_path, secondary_stem_path = separator.separate()
        primary_stem_path, secondary_stem_path = os.path.join(
            output_path, primary_stem_path), os.path.join(output_path, secondary_stem_path)

        # Load the secondary stem, set channels to 1, and export as WAV
        sound = AudioSegment.from_wav(secondary_stem_path)
        sound = sound.set_channels(1)
        sound.export(secondary_stem_path, format="wav")
        print(Fore.BLUE+'Аудио-файл разделён успешно!'+Fore.WHITE)
        return primary_stem_path, secondary_stem_path
    except Exception as e:
        print(Fore.RED+f"Ошибка при разделении аудио: {e}")
        return None


def speech_recognize(*, audio_path: str) -> Optional[str]:
    """
    Распознает речь в аудио-файле с использованием модели для распознавания речи.

    Parameters:
        audio_path (str): Путь к аудио-файлу.

    Returns:
        str: Распознанный текст.
    """
    global model_sr
    print(Fore.BLUE+"Распознавание речи..."+Fore.WHITE)
    try:
        result = model_sr.transcribe(audio_path, task='translate')
        if isinstance(result['text'], str):
            recognized_text: str = result['text']
            print(Fore.BLUE+'Речь успешно распознана!'+Fore.WHITE)
            return recognized_text
    except Exception as e:
        print(Fore.RED+f"Ошибка при распознавании речи: {e}")
        return None


def translate_text(*, text: str, target_language: Langs) -> Optional[str]:
    """
    Переводит текст на указанный язык с использованием библиотеки для перевода.

    Parameters:
        text (str): Текст для перевода.
        target_language (Literal['en', 'fr']): Язык, на который нужно перевести (en - английский, fr - французский).

    Returns:
        str: Переведенный текст.
    """
    print(Fore.BLUE+"Перевод..."+Fore.WHITE)
    source_language = "en"  # Исходный язык текста
    try:
        if (target_language == 'en'):
            print(Fore.BLUE+'Речь успешно переведена!'+Fore.WHITE)
            return text
        # Обновление пакетов для перевода
        arg_package.update_package_index()

        # Получение доступных пакетов
        available_packages = arg_package.get_available_packages()

        # Фильтрация пакетов для установки
        package_to_install = next(
            filter(
                lambda x: x.from_code == source_language and x.to_code == target_language,
                available_packages
            )
        )

        # Установка пакета
        arg_package.install_from_path(package_to_install.download())

        # Перевод текста
        translated_text = arg_translate.translate(
            text, source_language, target_language)
        print(Fore.BLUE+'Речь успешно переведена!'+Fore.WHITE)
        return translated_text
    except Exception as e:
        print(Fore.RED+f"Ошибка при переводе текста:", e)
        return None


def synthesize_speech(*, text: str, file_name: str, output_path: str, lang: Langs) -> Optional[str]:
    """
    Синтезирует речь из текста и сохраняет в аудио-файл.

    Parameters:
        text (str): Текст для синтеза речи.
        file_name (str): Название файла для сохранения.
        output_path (str): Путь для сохранения аудио-файла.
        lang (Langs): Язык синтеза речи.

    Returns:
        str: Путь к созданному аудио-файлу.
    """
    global device  # Предполагается, что переменная `device` определена где-то в коде.

    print(Fore.BLUE+"Синтезирование речи..."+Fore.WHITE)
    out_mp3_path = os.path.join(output_path, 'out_mp3')

    try:
        if os.path.exists(out_mp3_path):
            shutil.rmtree(out_mp3_path)

        os.makedirs(out_mp3_path, exist_ok=True)

        speaker = Speaker(model_id=f"v3_{lang}", language=lang,  # type: ignore
                          speaker=f"{lang}_1", device=device, logging=False)  # type: ignore
        audio_file_path: str = speaker.to_mp3(
            text=text, name_text=file_name, sample_rate=48000, audio_dir=out_mp3_path, speed=1.0)  # type: ignore

        cache_path = os.path.join(out_mp3_path, 'cache')
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)

        print(Fore.BLUE+'Речь успешно синтезирована!'+Fore.WHITE)
        return audio_file_path

    except Exception as e:
        print(Fore.RED+f"Ошибка при синтезе речи: {e}")
        return None


def load_models() -> bool:
    """
    Загружает модели и устанавливает глобальные переменные.

    Returns:
        bool: True, если загрузка успешна, False в противном случае.
    """
    global model_sr, device

    try:
        loaded_model_sr = whisper.load_model(
            "medium", device=device, download_root=path_to('models', 'openai'))

        # Если загрузка успешна, присваиваем глобальной переменной model_sr
        model_sr = loaded_model_sr
        return True

    except Exception as e:
        print(Fore.RED+f"Ошибка при загрузке модели: {e}")
        return False

# Сборка файлов в видео


def sync_non_speech_fragments(*, origin_file: str, target_file: str) -> bool:
    """
    Синхронизирует аудио-файлы, удаляя фрагменты без речи из целевого файла.

    Parameters:
        origin_file (str): Путь к исходному аудио-файлу с речью.
        target_file (str): Путь к целевому аудио-файлу, который будет модифицирован.

    Returns:
        bool: True, если операция завершена успешно, False в случае ошибки.
    """

    print(Fore.BLUE+"Синхронизация фрагментов..."+Fore.WHITE)
    try:
        # Загружаем аудио-файлы
        sound2 = AudioSegment.from_file(target_file)
        original_vocal = AudioSegment.from_file(origin_file)

        # Используем библиотеку silence для обнаружения фрагментов без речи
        non_speech_fragments = detect_silence(
            original_vocal, min_silence_len=1000, silence_thresh=-50)

        # Обработка каждого фрагмента без речи
        for fragment in non_speech_fragments:
            # Разбиваем звуковой сегмент перед фрагментом, добавляем фрагмент без речи, затем оставшуюся часть сегмента
            s1 = sound2[:fragment[0]]
            s2 = sound2[fragment[0]:]
            sound2 = s1 + original_vocal[fragment[0]:fragment[1]] + s2

        # Экспортируем измененный звуковой сегмент
        sound2.export(target_file, format='mp3')

        print(Fore.BLUE+'Фрагменты успешно синхронизированы!'+Fore.WHITE)
        return True

    except Exception as e:
        print(Fore.RED+f"Ошибка при синхронизации фрагментов: {e}")
        return False


def combine_audio_files(*, background_file: str, voice_file: str, output_path: str) -> Optional[str]:
    """
    Комбинирует аудио-файлы фонового звука и речи.

    Parameters:
        background_file (str): Путь к аудио-файлу фонового звука.
        voice_file (str): Путь к аудио-файлу с речью.
        output_path (str): Путь для сохранения комбинированного аудио-файла.

    Returns:
        Optional[str]: Путь к созданному комбинированному аудио-файлу, или None в случае ошибки.
    """
    print(Fore.BLUE+"Объединение синтезированной речи и фоновых звуков..."+Fore.WHITE)
    try:
        # Загружаем аудио-файлы
        sound1 = AudioSegment.from_file(background_file)
        sound2 = AudioSegment.from_file(voice_file)

        # Комбинируем фоновый звук с речью
        combined = sound1.overlay(sound2)

        # Создаем путь для сохранения комбинированного аудио-файла
        combined_file_path = os.path.join(
            output_path, 'combined.translated.mp3')

        # Экспортируем комбинированный аудио-файл
        combined.export(combined_file_path, format='mp3')

        print(Fore.BLUE+'Аудио-файлы успешно объединены!'+Fore.WHITE)
        return combined_file_path

    except Exception as e:
        print(Fore.RED+f"Ошибка при объединении аудио-файлов: {e}")
        return None


def replace_audio_in_video(*, audio_file: str, video_file: str) -> Optional[str]:
    """
    Заменяет аудио-дорожку в видео-файле на указанный аудио-файл.

    Parameters:
        audio_file (str): Путь к аудио-файлу, который будет использован для замены.
        video_file (str): Путь к видео-файлу, в котором нужно заменить аудио.

    Returns:
        Optional[str]: Путь к созданному видео-файлу с замененной аудио-дорожкой, или None в случае ошибки.
    """
    print(Fore.BLUE+"Замена аудио в видео..."+Fore.WHITE)
    try:
        # Определяем путь для сохранения результата
        result_path = '.'.join(os.path.basename(
            video_file).split('.')[:-1]) + '.translated.mp4'

        # Выполняем команду ffmpeg для замены аудио в видео
        subprocess.call(['ffmpeg', '-i', video_file, '-i', audio_file, '-c:v', 'copy',
                         '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-y', result_path, '-loglevel', 'quiet'])

        return result_path

    except Exception as e:
        print(Fore.RED+f"Ошибка при замене аудио в видео: {e}")
        return None


def main(*, input_path: str, output_path: str, from_language: Literal['ru'] = 'ru', to_language: Langs) -> str:
    if not load_models():
        return Fore.RED+"Ошибка: Модель искусственного интеллекта не может быть загружена."

    audio_path = get_audio(input_path=input_path, output_dir=output_path)
    if audio_path is None:
        return Fore.RED+"Ошибка: Путь к файлу не найден."

    split_result = split_audio(input_path=audio_path, output_path=output_path)
    if split_result is None:
        return Fore.RED+"Ошибка: Видео не удалось разделить."

    background, voice = split_result
    recognized_text = speech_recognize(audio_path=voice)
    if recognized_text is None:
        return Fore.RED+"Ошибка: Текст не удалось распознать."

    translated_text = translate_text(
        text=recognized_text, target_language=to_language)
    if translated_text is None:
        return Fore.RED+"Ошибка: Текст перевода пуст."

    synth_file = synthesize_speech(text=translated_text, file_name='.'.join(
        os.path.basename(input_path).split('.')[:-1]), output_path=output_path, lang=to_language)
    if synth_file is None:
        return Fore.RED+"Ошибка: Синтез речи завершился с ошибкой."

    synced_audio = sync_non_speech_fragments(
        origin_file=voice, target_file=synth_file)
    if not synced_audio:
        return Fore.RED+"Ошибка: Файл не удалось синхронизировать."

    combined_audio = combine_audio_files(
        background_file=background, voice_file=synth_file, output_path=output_path)
    if combined_audio is None:
        return Fore.RED+"Ошибка: Не удалось объединить видео с аудио."

    translated_video = replace_audio_in_video(
        audio_file=combined_audio, video_file=input_path)
    if translated_video is None:
        return Fore.RED+"Ошибка: Видео не удалось перевести."

    return Fore.GREEN+f"Успех: Видео успешно переведено с {from_language} на {to_language}! Сохранено по пути {translated_video}"


if __name__ == '__main__':
    os.system('cls||clear')
    print(Fore.BLUE+"Начало работы программы"+Fore.WHITE)

    parser = ArgumentParser(prog="video-translator")
    parser.add_argument('--input', type=str,
                        help="полный путь до видео которое нужно перевести")
    parser.add_argument(
        '--output', type=str, help="полный путь до папки в которой будут расположены все выходные данные")
    parser.add_argument(
        '--target-language', type=str, required=False, choices=['en', 'fr'], default='en', help="код языка на который нужно перевести видео (по умолчанию - en)")
    parser.add_argument(
        '--initial-language', type=str, required=False, default='ru', help="код языка в видео (по умолчанию - ru)")
    parser.add_argument(
        '--use-cuda', type=bool, required=False, default=False, help="использовать cuda или нет (В зависимости от установки)")

    args = parser.parse_args()
    device = 'cuda' if args.use_cuda else 'cpu'

    message = main(input_path=args.input, output_path=args.output,
                   from_language=args.initial_language, to_language=args.target_language)

    print(
        Fore.BLUE+f"Работы программы завершена. Результат выполнения программы: \n\t{message}")
