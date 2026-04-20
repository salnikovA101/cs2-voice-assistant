# 🎮 CS2 Voice Assistant

**Голосовой ИИ-ассистент для Counter-Strike 2**  
Нажимаешь кнопку → говоришь → ассистент **видит твой экран** и отвечает голосом в реальном времени.

**Всё зависит от промпта и модели LLM**  
При правильном промпте + быстрой модели (например Gemini 3.1 Flash Lite) весь пайплайн укладывается в **~3 секунды**.  
На более «тяжёлых» моделях — до 15–20 секунд.  

**Основные сценарии:**
- Юмористический трэш-ток и троллинг тиммейтов
- Реальная помощь в игре (экономика, позиции, тайминги, call’ы)

---

## ✨ Возможности

- **Видит экран** — DXCamera + Gemini Vision
- **Русская речь** — Faster-Whisper (отлично понимает CS2-сленг)
- **Два режима TTS**:
  - **Speed** — Silero V5 (быстрый отклик)
  - **Quality** — FasterQwen3TTS с клонированием любого твоего голоса
- **Два стиля ответов** (меняются в `config.yaml`):
  - `smart` — полезный анализ
  - `humor` — юмор и угар
- **История диалога** (настраивается)
- **Push-to-Talk** (любая клавиша)
- Работает **поверх игры**, без оверлеев

---

## 🚀 Установка

### 1. Клонируй репозиторий
```bash
git clone https://github.com/salnikovA101/cs2-voice-assistant
cd cs2-voice-assistant
```

### 2. Создай виртуальное окружение
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Установи зависимости
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### 4. Настрой API
Создай файл `.env` в корне проекта:
```env
API_KEY=AIzaSy...твой_Gemini_API_ключ
```

---

## ⚙️ Настройка (`config.yaml`)

```yaml
general:
  push_to_talk_key: "right ctrl"

stt: #~1,1гб VRAM
  model: "large-v3-turbo"
  device: "cuda"
  compute_type: "int8_bfloat16"

tts:
  mode: "speed" # "quality" - Qwen3 или "speed" - Silero v5
  # Qwen: ~2,4гб VRAM
  model: "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
  attn_implementation: "sdpa"
  max_seq_len: 1024
  voice: "voices/example.wav"
  # Silero ~0,2гб VRAM
  silero_speaker: "baya"
  sample_rate: 48000

llm: #~0гб VRAM
  model: "gemma-4-26b-a4b-it"
  api_key: ""
  temperature: 1.0
  max_output_tokens: 1000
  history_len: 3
  prompt_mode: "humor" # humor or smart
```

**Папка промптов** (`core/prompts/`):
- `smart.txt`
- `humor.txt`
- `silero_fix.txt` (автоматически подключается в speed-режиме)

---

## ▶️ Запуск

```bash
python main.py
```

**Как пользоваться:**
1. Запусти `main.py`
2. **Удерживай** кнопку PTT и говори
3. **Отпусти** кнопку — ассистент мгновенно отвечает голосом
4. При первом запуске скажет «Слушаю»

---

## 📁 Структура проекта

```
CS2-VOICE-ASSISTANT/
├── core/
│   ├── prompts/
│   │   ├── humor.txt
│   │   ├── silero_fix.txt
│   │   └── smart.txt
│   ├── assistant.py
│   └── context.py
├── providers/
│   ├── llm_provider.py
│   ├── ocr_provider.py
│   ├── stt_provider.py
│   ├── tts_fast_provider.py
│   └── tts_provider.py
├── utils/
│   └── audio_recorder.py
├── voices/                  # сюда клади свои референсные голоса
├── .env
├── .gitignore
├── config.yaml
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
```
