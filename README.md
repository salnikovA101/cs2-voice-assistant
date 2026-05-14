# 🎙️ Voice Assistant

> A modular, multimodal AI agent with real-time voice interaction, screen perception, emotional TTS, and autonomous tool calling.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ Features

- **🎤 Push-to-Talk Voice Input** — Записывает речь по нажатию клавиши (настраивается), транскрибирует через `Faster-Whisper`.
- **🧠 Universal LLM Integration** — Единый OpenAI-совместимый интерфейс для локальных (Ollama, LM Studio) и облачных (Gemini, OpenRouter, любой OpenAI-совместимый API) моделей. Профили переключаются в `config.yaml`.
- **🔊 Emotional TTS** — Продвинутый синтез речи через `Faster-Qwen3-TTS` в двух режимах: **emotional** (ассистент сам задаёт эмоцию через `<instruct>`) и **copy_base** (zero-shot клонирование голоса по референсу).
- **🖥️ Screen Perception** — Опциональный захват экрана через `DXcam` для передачи изображения в мультимодальный LLM.
- **🛠️ Autonomous Tool Calling** — Ассистент автономно выбирает и вызывает инструменты:
  - `launch_cs2` — запуск CS2 с проверкой свободной VRAM
  - `enable_game_mode` — переключение на легкий профиль LLM
  - `get_system_stats` — CPU/GPU/RAM телеметрия в реальном времени
  - `play_youtube_video` — поиск и запуск видео на YouTube
  - `web_search` — поиск в интернете без браузера, модель анализирует запрос и выдает ответ в текстовом формате
  - `set_timer` / `cancel_timer` — управление таймерами
- **⚡ Async Pipeline** — Весь цикл построен на `asyncio`, тяжёлые вычисления вынесены в потоки — минимальные задержки.

---

## 🏗️ Architecture

```text
cs2-voice-assistant/
├── .github/
│   └── workflows/
│       └── tests.yml           # CI: Автоматическое тестирование
├── core/
│   └── assistant.py            # Главный класс: оркестрация пайплайна
├── llm/
│   ├── base.py                 # BaseLLMProvider: запросы к llm и цикл tool calling
│   ├── history_manager.py      # Управление контекстом диалога
│   ├── manager.py              # Выбор провайдера по профилю
│   ├── prompt_loader.py        # Загрузка промптов из файлов
│   └── providers/
│       ├── lm_studio_provider.py
│       ├── ollama_provider.py
│       └── openai_provider.py  # OpenAI-compatible провайдер
├── tts/
│   ├── base.py                 # BaseTTSProvider: абстрактный класс для озвучки
│   ├── manager.py              # Выбор TTS провайдера
│   └── providers/
│       ├── copy_tts.py         # Клонирование голоса
│       ├── emotional_tts.py    # Эмоциональный синтез
│       └── speed_tts.py        # Ускоренный синтез для игр
├── providers/
│   ├── screen_capture.py       # DXcam: быстрый захват экрана
│   └── stt_provider.py         # Faster-Whisper: распознавание речи
├── tools/
│   ├── base.py                 # @tool декоратор + генерация JSON схем
│   ├── registry.py             # Автоматическая регистрация инструментов
│   └── implementations/
│       ├── game_mode.py        # Настройка игрового режима
│       ├── launch_cs2.py       # Запуск игры
│       ├── system_stats.py     # Мониторинг CPU/GPU/RAM
│       ├── timer.py            # Таймеры
│       ├── web_search.py       # Web поиск
│       └── youtube.py          # Управление музыкой/видео
├── utils/
│   ├── audio_recorder.py       # Запись микрофона (PTT/VAD)
│   ├── config.py               # Валидация конфигурации через Pydantic
│   └── constants.py            # Глобальные константы
├── prompts/
│   ├── logic.txt               # Промпт логики ассистента
│   ├── output_clone.txt        # Шаблоны для клонирования
│   ├── output_emotional.txt    # Шаблоны для эмоций
│   └── output_speed.txt        # Промпты для быстрой речи
├── tests/                      # Набор Unit-тестов
│   ├── test_config.py
│   ├── test_llm_base.py
│   ├── test_tool_decorator.py
│   └── test_tool_registry.py
├── voices/                     # Аудио-референсы (.wav)
├── config.yaml                 # Настройки (LLM, TTS, Hotkeys)
├── requirements.txt            # Зависимости проекта
├── .env                        # API ключи
├── .gitignore
├── LICENSE
├── README.md                   # Документация
└── main.py                     # Точка входа в приложение
```

---

## 💻 Tech Stack

| Категория | Библиотеки |
|---|---|
| **Core** | Python 3.13+, asyncio |
| **STT** | `faster-whisper` (large-v3-turbo, CUDA) |
| **LLM** | `openai` SDK (универсальный интерфейс) |
| **TTS** | `faster-qwen3-tts`, `silero` |
| **Vision** | `dxcam`, `opencv-python` |
| **Hardware** | `psutil`, `pynvml` |
| **Config** | `pydantic-settings` v2, `PyYAML` |

---

## ⚙️ Installation

### Prerequisites

- Python **3.13+**
- CUDA-compatible GPU (рекомендуется RTX серии)
- [Ollama](https://ollama.com/) — если используете локальные модели

### 1. Clone the repo

```bash
git clone https://github.com/salnikovA101/cs2-voice-assistant
cd cs2-voice-assistant
```

### 2. Install PyTorch (важно для RTX 50XX / Blackwell)

> [!WARNING]
> Для видеокарт **RTX 5000-серии (Blackwell)** установите сборку вручную:
>
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
> ```
>
> Для всех остальных GPU этот шаг можно пропустить.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Скопируйте `.env.example` в `.env` и заполните нужные ключи:

```bash
copy .env.example .env
```

```env
# Gemini (Google AI Studio)
LLM__PROFILES__GEMINI__API_KEY=your_gemini_api_key_here

# OpenRouter / Другой OpenAI-совместимый провайдер
LLM__PROFILES__OTHER__API_KEY=your_api_key_here
LLM__PROFILES__OTHER__BASE_URL=https://openrouter.ai/api/v1
```

> [!NOTE]
> Для локальных провайдеров (Ollama, LM Studio) API ключи не нужны.

### 5. Configure `config.yaml`

Откройте `config.yaml` и настройте профиль LLM и параметры под себя.

**Выбор активного профиля LLM:**
```yaml
llm:
  current_profile: "other"  # ollama | lm_studio | gemini | other
```

**Основные настройки:**
```yaml
general:
  push_to_talk_key: "right ctrl"  # Клавиша PTT
  image: false                    # Включить захват экрана (мультимодальность)
  enable_voice_output: true       # TTS
  enable_voice_input: true        # STT / Push-to-Talk
  enable_text_input: true         # Ввод текста в консоль

tts:
  mode: "emotional"  # emotional | copy_base | speed
```

**Профили LLM (примеры):**
```yaml
profiles:
  ollama:
    provider: "ollama"
    model: "qwen3.5:9b"
    base_url: "http://127.0.0.1:11434/v1"

  gemini:
    provider: "openai"
    model: "gemini-3.1-flash-lite"
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"

  other:
    provider: "openai"
    model: "deepseek/deepseek-v4-flash"
    # base_url и api_key берутся из .env
```

### 6. (Optional) Add voice reference

Для режима **`copy_base`** (клонирование голоса) положите референсный WAV-файл в папку `voices/` и укажите путь в `config.yaml`:

```yaml
tts:
  clone:
    ref_voice: "voices/your_voice.wav"
    ref_text: "Текст из вашей записи..."
```

### 7. Run

```bash
python main.py
```

---

## 🎮 Usage

| Действие | Что происходит |
|---|---|
| Зажать `Right Ctrl` (или другую клавишу PTT) | Начинается запись голоса |
| Отпустить | Аудио транскрибируется → отправляется в LLM → ответ озвучивается |
| Написать в консоль | Параллельный текстовый ввод (если включён `enable_text_input`) |
| Сказать «запусти CS2» | Ассистент проверит VRAM и запустит игру |
| Сказать «включи игровой режим» | LLM переключится на быстрый профиль |

---

## 📄 License

[MIT](LICENSE)