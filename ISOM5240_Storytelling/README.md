# Magic Picture Stories — ISOM5240 Individual Assignment

A Streamlit app for children aged 3–10. A child uploads a picture, and the app
describes it, writes a short story about it (50–100 words) and reads the story
aloud.

## How it works

| Step | Hugging Face pipeline | Model |
|------|----------------------|-------|
| 1. Picture → caption | `image-text-to-text` | [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) |
| 2. Caption → story | `text-generation` | [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| 3. Story → speech | `text-to-speech` | [`facebook/mms-tts-eng`](https://huggingface.co/facebook/mms-tts-eng) |

Design choices:

- **Instruction-tuned story model.** A system prompt tells Qwen2.5-Instruct to
  write gentle, happy stories with simple words for young children. A plain
  GPT-2 story model can't follow instructions like that.
- **50–100 words, guaranteed.** The prompt asks for 60–90 words. The app then
  cleans up the text, trims it to 100 words at a sentence boundary, and
  retries up to 3 times if the story is under 50 words.
- **Fits Streamlit Community Cloud.** The two larger models are loaded in
  bfloat16, about 2 GB in total. Each model is loaded once with
  `st.cache_resource` and shared across users and reruns.
- **Kid-friendly UI.** Large story text, step-by-step progress messages, an
  audio player and a button to save the audio.

## Files

- `app.py`: application source code (with comments)
- `requirements.txt`: Python dependencies
- `../.streamlit/config.toml`: turns off Streamlit's file watcher, which
  otherwise crawls every transformers module and floods the log

## Run locally

From the repository root:

```bash
pip install -r ISOM5240_Storytelling/requirements.txt
streamlit run ISOM5240_Storytelling/app.py
```

The first run downloads about 2 GB of model weights.

## Deploy on Streamlit Community Cloud

1. Sign in at <https://share.streamlit.io> with GitHub.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `ludanzhang54/MBA`, branch: the branch with these files,
   main file path: `ISOM5240_Storytelling/app.py`.
4. Under **Advanced settings**, choose **Python 3.12**.
5. Click **Deploy**. The first build and model download take several minutes.

## Notes

- `transformers` is pinned to **5.16.0**. Version 5.17.0 has a bug in the
  `text-to-speech` pipeline that crashes with tokenizer-only models like
  `facebook/mms-tts-eng`
  (`BatchEncoding.to() got an unexpected keyword argument 'dtype'`).
- From transformers v5, the old `image-to-text` pipeline task no longer
  exists. BLIP now runs through `image-text-to-text`.
