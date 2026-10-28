import datetime
import logging
import signal
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent / "src"))
from ragga import (  # noqa: E402
    Config,
    Encoder,
    Generator,
    MarkdownDataset,
    VectorDatabase,
    WebSearchRetriever,
)
from ragga.crafting.prompt import Llama2ChatPrompt, Phi2ChatPrompt, Phi2QAPrompt, TinyLlamaChatPrompt  # noqa: E402

REGISTERED_EXIT_CALLBACKS: list = []

def signal_handler(_, __) -> None:
    logging.info("Exiting...")
    for callback in REGISTERED_EXIT_CALLBACKS:
        callback()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def output_model_response_stream(model_response: str) -> None:
    sys.stdout.write(model_response)
    sys.stdout.flush()


NUM_GENERATIONS = 3

models: dict[str, dict] = {
    "phi-2": {
        "quant_bits": 8,
        "prompt": Phi2QAPrompt,
        "config_override": {
            "generator": {
                "llm_path": "./models/phi-2.Q8_0.gguf",
                # "hf_tokenizer": "microsoft/phi-2",
                "llama": False,
                "context_length": 2048,
                "autoflush": False,
            }
        }
    },
    "phi-2-chat": {
        "quant_bits": 8,
        "prompt": Phi2ChatPrompt,
        "config_override": {
            "generator": {
                "llm_path": "./models/phi-2.Q8_0.gguf",
                # "hf_tokenizer": "microsoft/phi-2",
                "llama": False,
                "context_length": 2048,
                "autoflush": False,
            }
        }
    },
    "llama2": {
        "quant_bits": 5,
        "prompt": Llama2ChatPrompt,
        "config_override": {
            "generator": {
                "llm_path": "./models/llama-2-7b-chat.Q5_K_M.gguf",
                # "hf_tokenizer": "KoboldAI/llama2-tokenizer",
                "llama": False,  # Llama2ChatPrompt emits the [INST] format itself
                "context_length": 2048, # supports 4096 but less is faster and consistent with others
                "autoflush": False,
            }
        }
    },
    "tinyllama": {
        "quant_bits": 8,
        "prompt": TinyLlamaChatPrompt,
        "config_override": {
            "generator": {
                "llm_path": "./models/tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
                # "hf_tokenizer": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "llama": False,  # Otherwise it will use the default llama prompt, which is slightly different
                "context_length": 2048,
                "autoflush": False,
            }
        }
    },
}

datasets: dict[str, dict] = {
    "CLE_Course": {
        "source": "https://github.com/MVS-99/CLE_Course",
        "config_override": {
            "dataset": {
                "path": "../ObsidianVaults/CLE_Course",
            }
        }
    },
    "Cybersecurity-Notes": {
        "source": "https://github.com/Twigonometry/Cybersecurity-Notes",
        "config_override": {
            "dataset": {
                "path": "../ObsidianVaults/Cybersecurity-Notes",
            }
        }
    },
    "FDA-Notes": {
        "source": "https://github.com/Vuenc/FDA-Notes",
        "config_override": {
            "dataset": {
                "path": "../ObsidianVaults/FDA-Notes",
            }
        }
    },
    "MathWiki": {
        "source": "https://github.com/zhaoshenzhai/MathWiki",
        "config_override": {
            "dataset": {
                "path": "../ObsidianVaults/MathWiki",
            }
        }
    },
}

eval_qa: dict[str, list[tuple[str, str]]] = {
    "general": [
        ("What is 1 + 1?", "2"),
        ("What is the capital of France?", "Paris"),
    ],
    "CLE_Course": [
        (
            "What are the operators?",
            "Multiplication, Division, Addition, Subtraction, Unary Minus, "
            "Remainder, Prefix and Postfix, Shortcut, Logical, Bitwise, Bit Shifting, Size"
        ),
        (
            "Which bitwise operators are there?",
            "& (ampersand) -> Bitwise conjunction (AND), "
            "| (bar) -> Bitwise disjunction (OR), "
            "~ (tilde) -> Bitwise negation (NOT), "
            "^ (caret) -> Bitwise exclusive disjunction (XOR)"
        ),

    ],
    "Cybersecurity-Notes": [
        (
            "What do most assessments include?",
            "enumeration, "
            "identifying running services & the purpose of the target, "
            "re-enumerating services, "
            "finding software and hardware versions, "
            "searching for exploits in found versions, "
            "collecting credentials, "
            "exploitation, "
            "enumeration (again), "
            "persistence, "
            "privilege escalation, "
            "enumeration (again), "
            "persistence (again)"
        ),
        (
            "What is CVE-2020-12271?",
            "Pre-Authentication SQL Injection, "
            "can allow exfiltration of XG firewall-resident data, which can contain local user credentials"
        ),
    ],
    "FDA-Notes": [
        (
            "How can SVD be applied for Document Ranking?",
            "Represent documents as vectors that count word occurrences (say, the 25000 most important words "
            "in the English language). A collection of documents is then represented as a matrix. How to measure "
            "the intrinsic relevance of a document to the collection"
        ),
    ],
    "MathWiki": [
        (
            "What is the definition of an Exact Sequence?",
            r"Let $G_1,\dots,G_n$ be groups. A sequence $G_1\to G_2\to\cdots\to G_n$ of homomorphisms "
            r"$\phi_i:G_i\to G_{i+1}$ is said to beexact if $\im\phi_i=\ker\phi_{i+1}$ for all $i$. "
            r"A short exact sequence is an exact sequence $1\to A\overset{\phi}{\to}B\overset{\psi}{\to}C\to1$; that "
            r"is, a sequence for which $\phi$ is injective, $\psi$ is surjective, and $\im\phi=\ker\psi$. "
            "A long exact sequence is an exact sequence that is not short."
        ),
    ],
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("loading config...")
    conf = Config(config_path=Path(__file__).parent / "_config.yaml")
    report: pd.DataFrame = pd.DataFrame(columns=[
        "model",
        "dataset",
        "model_config",
        "command_kw",
        "documents",
        "full_context",
        "question",
        "reference_answer",
        "response",
        "stdout",
        "stderr"
    ])

    def save_report() -> None:
        ymd = datetime.datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d_%H%M%S")
        report.to_csv(f"./reports/report_{ymd}.csv", index=False)

    REGISTERED_EXIT_CALLBACKS.append(save_report)

    try:
        for dataset_name, dataset_conf in datasets.items():
            log_prefix = f"[DS:{dataset_name}] "
            logging.debug(f"Loading dataset: {dataset_name}")
            # update dataset config
            conf.merge(dataset_conf["config_override"])
            # set encoder cache_dir to dataset name, this is global across all models for this dataset
            conf["encoder"]["cache_dir"] = f".cache/{dataset_name}"
            # same for FAISS db cache
            conf["retriever"]["cache_dir"] = f".cache/faiss_{dataset_name}"


            logging.debug(f"{log_prefix}loading dataset, this will take a while the first time...")
            dataset = MarkdownDataset(conf)
            for model_name, model_conf in models.items():
                log_prefix = f"[DS:{dataset_name}][M:{model_name}] "
                logging.debug(f"Loading model: {model_name}")
                if "config_override" in model_conf:
                    conf.merge(model_conf["config_override"])
                prompt = model_conf["prompt"](conf)
                logging.debug(f"{log_prefix}loading encoder...")
                encoder = Encoder(conf)
                logging.debug(f"{log_prefix}loading faiss db...")
                faiss_db = VectorDatabase(conf, encoder)
                if not faiss_db.loaded_from_disk:
                    faiss_db.documents = dataset.documents
                logging.debug(f"{log_prefix}loading generator and model...")
                generator = Generator(conf, prompt, faiss_db, websearch=WebSearchRetriever)
                # generator.subscribe(output_model_response_stream)
                # clear llama model info
                stdout, stderr = generator.flush_stdout_stderr()
                for question, answer in eval_qa["general"]:
                    log_prefix = f"[DS:{dataset_name}][M:{model_name}][GQ:{question}] "
                    logging.debug(f"Question: {question}")
                    for i in range(NUM_GENERATIONS):
                        logging.debug(f"{log_prefix}Generation {i} of {NUM_GENERATIONS}")
                        response = generator.get_answer(question)
                        stdout, stderr = generator.flush_stdout_stderr()
                        report = pd.concat(
                            [
                                pd.DataFrame(
                                    [[
                                        model_name,
                                        dataset_name,
                                        model_conf,
                                        generator.last_keywords,
                                        generator.last_docs,
                                        generator.last_context,
                                        question,
                                        answer,
                                        response,
                                        stdout,
                                        stderr
                                    ]],
                                    columns=report.columns
                                ),
                                report
                            ],
                            ignore_index=True
                        ) # type: ignore
                if dataset_name not in eval_qa:
                    continue
                for question, answer in eval_qa[dataset_name]:
                    log_prefix = f"[DS:{dataset_name}][M:{model_name}][SQ:{question}] "
                    logging.debug(f"Question: {question}")
                    for i in range(NUM_GENERATIONS):
                        logging.debug(f"{log_prefix}Generation {i+1} of {NUM_GENERATIONS}")
                        response = generator.get_answer(question)
                        stdout, stderr = generator.flush_stdout_stderr()
                        report = pd.concat(
                            [
                                pd.DataFrame(
                                    [[
                                        model_name,
                                        dataset_name,
                                        model_conf,
                                        generator.last_keywords,
                                        generator.last_docs,
                                        generator.last_context,
                                        question,
                                        answer,
                                        response,
                                        stdout,
                                        stderr
                                    ]],
                                    columns=report.columns
                                ),
                                report
                            ],
                            ignore_index=True
                        )  # type: ignore
    except Exception as e:
        logging.exception(e)
        save_report()
        raise e

    save_report()
    logging.info("Done!")




