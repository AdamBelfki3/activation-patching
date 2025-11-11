from nnsight import LanguageModel
import nnsight
from rich.table import Table
from rich.console import Console
import torch

def load_model(model_name, remote=True):

    if model_name == "openai-community/gpt2-xl":
        model = LanguageModel(model_name, rename={"transformer": "model", "h": "layers"}, device_map="auto", dispatch=True)

        return model

    if not nnsight.is_model_running(model_name):
        print(nnsight.ndif_status())
        raise ValueError(f"Model {model_name} is not currently available on NDIF")

    model = LanguageModel(model_name, device_map="auto", dispatch=not remote)

    return model


def show_token_positions(str_tokens, title):


    table = Table(title=title, show_header=False)

    table.add_row(*str_tokens)
    table.add_row(*[str(i) for i in range(len(str_tokens))])
    table.rows[0].style = "bold"

    console = Console()
    console.print(table)

    return str_tokens


def show_patch_pattern(patch_position, str_tokens, title):

    table = Table(title=title, show_header=False)

    # table.add_row(*[f"[on purple]{t}[/on purple]" if i == patch_position else t for i, t in enumerate(str_tokens)])
    table.add_row(*str_tokens)
    table.add_row(*[str(i) for i in range(len(str_tokens))])

    table.rows[0].style = "bold"
    table.columns[patch_position].style = "on purple"

    console = Console()
    console.print(table)


def tokenize_prompt(prompt, model, show=False, title=None):

    str_tokens = model.tokenizer.batch_decode(model.tokenizer.encode(prompt))

    if show:
        show_token_positions(str_tokens, title)

    return str_tokens


def get_token_id(token_string, model):
    return model.tokenizer.encode(token_string, add_special_tokens=False)[0]
