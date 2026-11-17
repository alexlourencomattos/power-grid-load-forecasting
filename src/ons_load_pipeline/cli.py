import argparse
import logging
from datetime import datetime
from pathlib import Path

from .config import Settings
from .pipeline import run_pipeline

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extrai e trata a carga horária do ONS")
    parser.add_argument("--start-year", type=int, required=True, help="Primeiro ano a processar")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="Último ano")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Diretório de dados")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = run_pipeline(args.start_year, args.end_year, Settings(data_dir=args.data_dir))
    LOGGER.info("Execução %s concluída; %s linhas gold", manifest["run_id"], manifest["gold_rows"])


if __name__ == "__main__":
    main()

