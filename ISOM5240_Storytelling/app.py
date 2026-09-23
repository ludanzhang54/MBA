"""
ISOM5240 Individual Assignment - Storytelling Application
=========================================================

A Streamlit web app for kids aged 3-10. The user uploads a picture and the
app turns it into a short spoken story in three stages, each powered by a
pre-trained Hugging Face Transformers pipeline:

    1. Image captioning  (image -> caption)   Salesforce/blip-image-captioning-base
    2. Story generation  (caption -> story)   Qwen/Qwen2.5-0.5B-Instruct
    3. Text-to-speech    (story -> audio)     facebook/mms-tts-eng

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py
"""

import html
import io
import re
import wave

import numpy as np
import streamlit as st
import torch
from PIL import Image
from transformers import pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Hugging Face model IDs used by each stage of the app.
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TTS_MODEL = "facebook/mms-tts-eng"

# BLIP works best with a short text prefix; the caption continues it.
CAPTION_PROMPT = "a picture of"

# The assignment asks for a story of 50-100 words.
MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100

# Maximum number of attempts to get a story inside the word range.
MAX_STORY_ATTEMPTS = 3

# Instructions that keep the story short, simple and suitable for young kids.
STORY_SYSTEM_PROMPT = (
    "You are a kind and cheerful storyteller for children aged 3 to 10. "
    "Write stories with simple words and short sentences. "
    "Stories must be happy, gentle and safe: no violence, no scary monsters, "
    "no sad endings. Always end with a happy ending."
)

# Image formats accepted by the file uploader.
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


# ---------------------------------------------------------------------------
# Model loading
# st.cache_resource loads each model once per server process and reuses it for
# every user and every rerun, so the app doesn't reload ~2 GB of weights
# each time a button is clicked.
# ---------------------------------------------------------------------------

# All models use bfloat16 where possible: it halves memory use compared with
# float32, so everything fits within Streamlit Community Cloud's ~2.7 GB limit.

@st.cache_resource(show_spinner="Waking up the picture reader...")
def load_caption_pipeline():
    """Load the BLIP image-captioning pipeline."""
    return pipeline("image-text-to-text", model=CAPTION_MODEL, dtype=torch.bfloat16)


@st.cache_resource(show_spinner="Waking up the storyteller...")
def load_story_pipeline():
    """Load the instruction-tuned text-generation pipeline for the story."""
    return pipeline("text-generation", model=STORY_MODEL, dtype=torch.bfloat16)


@st.cache_resource(show_spinner="Warming up the story voice...")
def load_tts_pipeline():
    """Load the MMS text-to-speech pipeline (English voice).

    This model is small (~150 MB), so it stays in float32 for best audio quality.
    """
    return pipeline("text-to-speech", model=TTS_MODEL)


# ---------------------------------------------------------------------------
# Stage 1: image -> caption
# ---------------------------------------------------------------------------

def generate_caption(image):
    """Describe the uploaded image in one short sentence.

    Args:
        image: a PIL image uploaded by the user.

    Returns:
        The caption as a string, e.g. "a dog playing in the park".
    """
    captioner = load_caption_pipeline()
    # BLIP expects 3-channel RGB input; PNGs can be RGBA or palette-based.
    rgb_image = image.convert("RGB")
    result = captioner(images=rgb_image, text=CAPTION_PROMPT, max_new_tokens=30)
    caption = result[0]["generated_text"].strip()

    # BLIP repeats the prompt at the start of its output; remove it.
    if caption.lower().startswith(CAPTION_PROMPT):
        caption = caption[len(CAPTION_PROMPT):].strip()
    return caption


# ---------------------------------------------------------------------------
# Stage 2: caption -> story
# ---------------------------------------------------------------------------

def count_words(text):
    """Return the number of words in the text."""
    return len(text.split())


def clean_story(text):
    """Tidy up raw model output so it reads as plain story text.

    Removes markdown symbols (e.g. **bold**), a leading "Title:" line and
    extra whitespace, all of which would sound odd when read aloud.
    """
    text = text.replace("*", "").replace("#", "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Drop a leading title line such as "Title: The Happy Dog".
    if len(lines) > 1 and lines[0].lower().startswith("title"):
        lines = lines[1:]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def trim_to_word_limit(text, max_words=MAX_STORY_WORDS):
    """Shorten a story to at most max_words words, ending on a full sentence.

    If the text is already short enough it is returned unchanged. Otherwise
    it is cut to max_words and then back to the last sentence ending, so
    the story never stops mid-sentence.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    shortened = " ".join(words[:max_words])
    last_sentence_end = max(shortened.rfind("."), shortened.rfind("!"), shortened.rfind("?"))
    if last_sentence_end > 0:
        return shortened[:last_sentence_end + 1]
    return shortened + "."


def build_story_messages(caption):
    """Build the chat messages that ask the model for a kid-friendly story."""
    user_prompt = (
        f"Write a short story for young children about this picture: {caption}. "
        f"The story must be between 60 and 90 words long. "
        f"Give the main character a fun name. "
        f"Write only the story, without a title."
    )
    return [
        {"role": "system", "content": STORY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_story(caption):
    """Turn a caption into a 50-100 word children's story.

    The model is sampled up to MAX_STORY_ATTEMPTS times. The first story
    inside the 50-100 word range is returned. If none fits, the longest
    attempt is used, trimmed to the 100-word limit.

    Args:
        caption: the image caption from generate_caption().

    Returns:
        The story as a single string.
    """
    storyteller = load_story_pipeline()
    messages = build_story_messages(caption)
    best_story = ""

    for _ in range(MAX_STORY_ATTEMPTS):
        result = storyteller(
            messages,
            max_new_tokens=180,      # ~130 English words; enough for 100 words
            do_sample=True,          # sampling gives a different story each time
            temperature=0.8,
            top_p=0.9,
            return_full_text=False,  # return only the new text, not the prompt
        )
        story = trim_to_word_limit(clean_story(result[0]["generated_text"]))

        if MIN_STORY_WORDS <= count_words(story) <= MAX_STORY_WORDS:
            return story
        if count_words(story) > count_words(best_story):
            best_story = story

    return best_story


# ---------------------------------------------------------------------------
# Stage 3: story -> audio
# ---------------------------------------------------------------------------

def waveform_to_wav_bytes(waveform, sampling_rate):
    """Convert a float waveform (values in [-1, 1]) to WAV file bytes.

    WAV bytes can be played by st.audio and saved with st.download_button.
    """
    samples = np.clip(np.asarray(waveform, dtype=np.float32).squeeze(), -1.0, 1.0)
    pcm_samples = (samples * 32767).astype(np.int16)  # 16-bit PCM

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)          # mono
        wav_file.setsampwidth(2)          # 2 bytes = 16 bits per sample
        wav_file.setframerate(int(sampling_rate))
        wav_file.writeframes(pcm_samples.tobytes())
    return buffer.getvalue()


def text_to_speech(story):
    """Read the story aloud and return the audio as WAV bytes."""
    narrator = load_tts_pipeline()
    speech = narrator(story)
    return waveform_to_wav_bytes(speech["audio"], speech["sampling_rate"])


# ---------------------------------------------------------------------------
# User interface
# ---------------------------------------------------------------------------

def show_sidebar():
    """Show simple how-to instructions for kids and their parents."""
    with st.sidebar:
        st.header("How to play")
        st.markdown(
            "1. 📸 **Upload** a picture\n"
            "2. ✨ Press **Tell me a story!**\n"
            "3. 🎧 **Listen** to your story\n"
        )
        st.divider()
        st.caption(
            "Grown-ups: the first story takes a little longer while the "
            "AI models load. Stories are made by AI, so please read along "
            "with your child."
        )


def reset_story_if_new_image(uploaded_file):
    """Forget the previous story when a different picture is uploaded."""
    if st.session_state.get("image_id") != uploaded_file.file_id:
        st.session_state.image_id = uploaded_file.file_id
        st.session_state.pop("result", None)


def create_story(image):
    """Run the full image -> caption -> story -> audio process.

    Shows progress for each step and saves the result in st.session_state
    so it stays on screen when Streamlit reruns the script.
    """
    with st.status("Making your story...", expanded=True) as status:
        st.write("👀 Looking at your picture...")
        caption = generate_caption(image)

        st.write("✏️ Writing your story...")
        story = generate_story(caption)

        st.write("🎙️ Recording the story voice...")
        audio_bytes = text_to_speech(story)

        status.update(label="Your story is ready!", state="complete", expanded=False)

    st.session_state.result = {"caption": caption, "story": story, "audio": audio_bytes}


def show_story(result):
    """Display the caption, the story text and the audio player."""
    st.info(f"🔍 I can see **{result['caption']}**")

    st.subheader("📖 Your Story")
    # Larger text is easier for young readers. The story is escaped because it
    # is model output being inserted into HTML.
    st.markdown(
        f"<div style='font-size:1.25rem; line-height:1.7;'>{html.escape(result['story'])}</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"{count_words(result['story'])} words")

    st.subheader("🎧 Listen to Your Story")
    st.audio(result["audio"], format="audio/wav")
    st.download_button(
        "⬇️ Save the story audio",
        data=result["audio"],
        file_name="my_story.wav",
        mime="audio/wav",
    )


def main():
    """Entry point: lay out the page and handle the user's actions."""
    st.set_page_config(page_title="Magic Picture Stories", page_icon="🧚", layout="centered")
    st.title("🧚 Magic Picture Stories")
    st.write("Show me a picture and I will tell you a story about it!")
    show_sidebar()

    uploaded_file = st.file_uploader("📸 Choose a picture", type=ALLOWED_IMAGE_TYPES)
    if uploaded_file is None:
        st.session_state.pop("result", None)
        return

    reset_story_if_new_image(uploaded_file)

    try:
        image = Image.open(uploaded_file)
    except Exception:
        st.error("Oops! I couldn't open that picture. Please try another one.")
        return
    st.image(image, caption="Your picture", width=450)

    # Each click makes a new story, because the story model samples randomly.
    if st.button("✨ Tell me a story!", type="primary", width="stretch"):
        try:
            create_story(image)
        except Exception as error:
            # Show a friendly message instead of a stack trace.
            st.error("Oh no! Something went wrong while making your story. Please try again.")
            st.exception(error)
            return

    if "result" in st.session_state:
        show_story(st.session_state.result)


if __name__ == "__main__":
    main()
