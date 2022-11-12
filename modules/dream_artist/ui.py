import html

import gradio as gr

import modules.dream_artist.cptuning
import modules.dream_artist.preprocess
from modules import sd_hijack, shared


def create_embedding(name, initialization_text, nvpt, overwrite_old, use_negative, nvpt_neg):
    filename = modules.dream_artist.cptuning.create_embedding(name, nvpt, overwrite_old, init_text=initialization_text)
    if use_negative:
        modules.dream_artist.cptuning.create_embedding(name+'-neg', nvpt_neg, overwrite_old, init_text=initialization_text)
        filename=f'{filename} and {filename[:-3]}-neg.pt'

    sd_hijack.model_hijack.embedding_db.load_words_embeddings()

    return gr.Dropdown.update(choices=sorted(sd_hijack.model_hijack.embedding_db.word_embeddings.keys())), f"Created: {filename}", ""


def preprocess(*args):
    modules.dream_artist.preprocess.preprocess(*args)

    return "Preprocessing finished.", ""


def train_embedding(*args):

    assert not shared.cmd_opts.lowvram, 'Training models with lowvram not possible'

    apply_optimizations = shared.opts.training_xattention_optimizations
    try:
        if not apply_optimizations:
            sd_hijack.undo_optimizations()

        embedding, filename = modules.dream_artist.cptuning.train_embedding(*args)

        res = f"""
Training {'interrupted' if shared.state.interrupted else 'finished'} at {embedding.step} steps.
Embedding saved to {html.escape(filename)}
"""
        return res, ""
    except Exception:
        raise
    finally:
        if not apply_optimizations:
            sd_hijack.apply_optimizations()

