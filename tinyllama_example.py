"""Chat with your notes, using RAGGA as a library (for the command line tool, see `ragga --help`).

Usage: python tinyllama_example.py [model.gguf] [notes folder]
By default it uses the first .gguf file in ./models and the notes in ./data.
"""

import logging
import sys
from pathlib import Path

# The line below is only necessary if you are using RAGGA from a cloned repository
sys.path.append(str(Path(__file__).parent / "src"))

from ragga import (
    Config,
    Encoder,
    Generator,
    GGUFChatPrompt,
    MarkdownDataset,
    VectorDatabase,
    WebSearchRetriever,
)
from ragga.core.config import DEFAULT_CONFIG_PATH

HERE = Path(__file__).parent


def output_model_response_stream(model_response: str) -> None:
    sys.stdout.write(model_response)
    sys.stdout.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.debug("loading config...")
    conf = Config(config_path=DEFAULT_CONFIG_PATH)

    args = sys.argv[1:]
    models = sorted((HERE / "models").glob("*.gguf"))
    model = Path(args[0]) if args else (models[0] if models else None)
    notes = Path(args[1]) if args[1:] else HERE / "data"
    if model is None or not model.is_file():
        sys.exit(f"No model found: download a GGUF chat model (see README) into {HERE / 'models'}, or pass its path")
    if not notes.is_dir():
        sys.exit(f"Notes folder not found: {notes} (put markdown notes there, or pass a folder)")
    conf.merge({"generator": {"llm_path": str(model)}, "dataset": {"path": str(notes)}})

    logging.info("loading dataset, this will take a while the first time...")
    dataset = MarkdownDataset(conf)

    logging.info("loading encoder...")
    encoder = Encoder(conf)
    logging.debug("loading vector db...")
    vector_db = VectorDatabase(conf, encoder)
    # always sync: the indexer only embeds new or changed notes, and drops deleted ones
    vector_db.documents = dataset.documents
    logging.debug("loading generator and model...")
    prompt = GGUFChatPrompt(conf)  # uses the chat template embedded in the model file
    logging.info("loading chatbot...")
    generator = Generator(conf, prompt, vector_db, websearch=WebSearchRetriever)
    generator.subscribe(output_model_response_stream)
    logging.info("chatbot ready!")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            logging.debug("user input: %s", user_input)
            model_response = generator.get_answer_stream(user_input)
            full_response = ""
            for response in model_response:
                full_response += response
            sources = dict.fromkeys(d.metadata["title"] for d in generator.last_docs or [] if "title" in d.metadata)
            print("\n\nSources: " + ", ".join(sources) if sources else "")  # noqa: T201
        except KeyboardInterrupt:
            break
