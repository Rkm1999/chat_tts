"""
main.py - Interactive streaming TTS terminal for Qwen3-TTS GGUF
Auto-starts with Aiden voice; supports clone mode and command interface.
"""
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen3_tts_gguf.inference import TTSEngine, TTSConfig, TTSResult
from qwen3_tts_gguf.inference.schema.constants import SPEAKER_MAP, LANGUAGE_MAP


def print_help():
    print("\n" + "=" * 55)
    print("  Qwen3-TTS Commands:")
    print("  /help                         show this help")
    print("  /speakers                     list built-in speakers")
    print("  /languages                    list supported languages")
    print("  /voice <name> <lang> <text>   synthesize & lock a new voice anchor")
    print("  /custom <text> <speaker> [instruct]  use built-in voice")
    print("  /design <text> <instruct>     design a voice from description")
    print("  /load <path>                  load voice anchor from JSON")
    print("  /save <path>                  save last result (.wav or .json)")
    print("  /temp <value>                 adjust sampling temperature")
    print("  /info                         show current engine state")
    print("  /reset                        clear memory and voice anchor")
    print("  /q, /exit                     quit")
    print("=" * 55)


def interactive_session():
    print("\nStarting Qwen3-TTS interactive terminal...")

    engine = TTSEngine(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'), verbose=False)
    stream = engine.create_stream()

    if stream is None:
        print("ERROR: Engine initialization failed. Check model file path.")
        return

    cfg = TTSConfig(temperature=0.6, sub_temperature=0.6, seed=42, sub_seed=45)
    last_result: Optional[TTSResult] = None

    # Auto-init Sohee voice
    print("\nInitializing Sohee voice anchor...")
    try:
        last_result = stream.set_voice('Sohee', text='안녕하세요, 저는 소희입니다.', language='korean')
        stream.join()
        print("Sohee voice locked. Type any text to speak, or /help for commands.")
    except Exception as e:
        print(f"WARNING: Failed to init Sohee voice: {e}")
        print("You can set a voice manually with /voice or /load.")

    print_help()

    try:
        while True:
            try:
                raw_input = input("\n[Sohee] >>> ").strip()
            except EOFError:
                break

            if not raw_input:
                continue

            if raw_input.startswith('/'):
                parts = raw_input.split(maxsplit=4)
                cmd = parts[0].lower()

                if cmd == '/help':
                    print_help()
                elif cmd == '/info':
                    print(f"\n  Temperature : {cfg.temperature}")
                    print(f"  Voice       : {stream.voice.info if stream.voice else 'not set'}")
                elif cmd == '/speakers':
                    names = sorted(SPEAKER_MAP.keys())
                    print("\nBuilt-in speakers:\n  " + ", ".join(names))
                elif cmd == '/languages':
                    langs = sorted(LANGUAGE_MAP.keys())
                    print("\nSupported languages:\n  " + ", ".join(langs))
                elif cmd == '/voice':
                    if len(parts) < 4:
                        print("Usage: /voice <name> <lang> <text>")
                        continue
                    spk, lang, v_text = parts[1], parts[2], parts[3]
                    print(f"Building voice anchor for [{spk}]...")
                    last_result = stream.set_voice(spk, text=v_text, language=lang)
                    stream.join()
                    print("Voice locked.")
                elif cmd == '/load':
                    if len(parts) < 2:
                        print("Usage: /load <path>")
                        continue
                    last_result = stream.set_voice(parts[1])
                    print(f"Voice loaded from: {parts[1]}")
                elif cmd == '/save':
                    if len(parts) < 2:
                        print("Usage: /save <filename>")
                        continue
                    if last_result:
                        last_result.save(parts[1])
                        print(f"Saved to: {parts[1]}")
                    else:
                        print("No result to save yet.")
                elif cmd == '/temp':
                    if len(parts) < 2:
                        print("Usage: /temp <value>")
                        continue
                    cfg.temperature = float(parts[1])
                    print(f"Temperature set to: {cfg.temperature}")
                elif cmd == '/design':
                    if len(parts) < 3:
                        print("Usage: /design <text> <instruct>")
                        continue
                    print("Designing voice and synthesizing...")
                    last_result = stream.design(parts[1], instruct=parts[2], config=cfg)
                    stream.join()
                elif cmd == '/custom':
                    if len(parts) < 3:
                        print("Usage: /custom <text> <speaker> [instruct]")
                        continue
                    ins = parts[3] if len(parts) > 3 else None
                    print(f"Synthesizing with built-in voice [{parts[2]}]...")
                    last_result = stream.custom(parts[1], speaker=parts[2], instruct=ins, config=cfg)
                    stream.join()
                elif cmd == '/reset':
                    stream.reset()
                    print("Memory and voice anchor cleared.")
                elif cmd in ['/q', '/exit', '/quit']:
                    break
                else:
                    print(f"Unknown command: {cmd}  (type /help for list)")
                continue

            # Standard synthesis in clone mode
            if not stream.voice:
                print("No voice anchor set. Use /voice, /load, /custom, or /design first.")
                continue

            try:
                print("Synthesizing...")
                last_result = stream.clone(raw_input, config=cfg)
                stream.join()
                if last_result:
                    if last_result.rtf == 0.0:
                        print("WARNING: No audio generated (RTF 0.00). The model may not handle this word/sound.")
                        print("  Try adding more context, e.g. a full sentence.")
                    else:
                        print(f"Done. [RTF: {last_result.rtf:.2f}]")
            except RuntimeError as e:
                print(f"Error: {e}")

    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"\nRuntime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.shutdown()


if __name__ == "__main__":
    interactive_session()
