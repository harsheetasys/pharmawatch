"""One-command pipeline: ingest -> load DB -> train model -> build RAG index.

    python -m scripts.build_all                       # synthetic world (offline)
    python -m scripts.build_all --source openfda      # real FDA data (needs internet)
"""
from __future__ import annotations

import argparse
import time

from pharmawatch import config, db, model, rag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=config.DATA_MODE, choices=["synthetic", "openfda"])
    ap.add_argument("--n", type=int, default=24000, help="synthetic report count")
    ap.add_argument("--per-drug", type=int, default=3000, help="openFDA reports per drug")
    args = ap.parse_args()

    t0 = time.time()
    if args.source == "synthetic":
        from pharmawatch.ingest import synthetic
        frames = synthetic.generate(args.n)
    else:
        from pharmawatch.ingest import openfda
        frames = openfda.build(per_drug=args.per_drug)
    print({k: len(v) for k, v in frames.items()})

    engine = db.get_engine()
    db.reset_schema(engine)
    db.load_frames(engine, frames)
    print(f"[db] loaded into {engine.dialect.name} in {time.time() - t0:.1f}s")

    model.train(engine)
    n = rag.build_index(engine)
    print(f"[rag] indexed {n} label chunks | total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
