# **Перевод видео с Rutube на другие языки**



![Картинка](./assets/image.png)


## Установка

1. Склонируйте репозиторий и перейдите в папку со склонированным репозиторием
    ```bash
    git clone https://github.com/sekrittt/Rutube_hack_2023.git
    cd Rutube_hack_2023
    ```
    Или скачайте zip-файл, распакуйте его и запустите в папке со склонированным репозиторием терминал

2. Создайте виртуальную среду:
    ```bash
    python -m venv venv
    ```
3. Активируйте виртуальную среду:
    - **На Windows**:

        ```bash
        venv\Scripts\activate
        ```

    - **На macOS и Linux**:

        ```bash
        source venv/bin/activate
        ```
4. Установите необходимые зависимости:
    - **Для работы на GPU**
    ```bash
    pip install -r requirements.gpu.txt
    ```
    - **Для работы на CPU**
    ```bash
    pip install -r requirements.txt
    ```
5. Установите программу ffmpeg

6. Программа готова к [использованию](#using)

## <a id="using">Использование</a>

Для использования нашей программы нужно выполнить следуюущую команду:
```bash
python main.py --input <path_to_video> --output <output_dir> --target-language <language_code> --initial-language <language_name> --use-cuda <need_cuda>
```
**path_to_video** - полный путь до видео которое нужно перевести <br>
**output_dir** - полный путь до папки в которой будут расположены все выходные данные <br>
**language_code** - код языка на который нужно перевести видео (по умолчанию - *en*)<br>
**language_name** - код языка в видео (по умолчанию - *ru*)<br>
**need_cuda** - использовать cuda или нет (*В зависимости от установки*)

## Используемые Технологии

| Название | Версия |
| --- | --- |
| Python | 3.11 |
| pydub | 0.25.1 |
| torch | 2.1.0+cu118 |
| nltk | 3.8.1 |
| scipy | 1.11.3 |
| onnxruntime | 1.16.2 |
| onnxruntime-gpu | 1.16.2 |
| sounddevice | 0.4.6 |
| silero | 0.4.1 |
| num2words | 0.5.13 |
| openai-whisper | 20231106 |
| argostranslate | 1.9.1 |
