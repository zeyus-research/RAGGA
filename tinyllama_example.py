import logging
import sys
from pathlib import Path

# The line below is only necessary if you are using RAGGA from a cloned repository
sys.path.append(str(Path(__file__).parent / "src"))

from ragga import (
    Config,
    Encoder,
    Generator,
    MarkdownDataset,
    VectorDatabase,
    WebSearchRetriever,
)
from ragga.crafting.prompt import TinyLlamaChatPrompt


def output_model_response_stream(model_response: str) -> None:
    sys.stdout.write(model_response)
    sys.stdout.flush()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("loading config...")
    conf = Config(config_path=Path(__file__).parent / "config.yaml")

    logging.info("loading dataset, this will take a while the first time...")
    dataset = MarkdownDataset(conf)

    logging.info("loading encoder...")
    encoder = Encoder(conf)
    logging.debug("loading vector db...")
    vector_db = VectorDatabase(conf, encoder)
    if not vector_db.loaded_from_disk:
        vector_db.documents = dataset.documents
    logging.debug("loading generator and model...")
    prompt = TinyLlamaChatPrompt(conf)
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
        except KeyboardInterrupt:
            break
