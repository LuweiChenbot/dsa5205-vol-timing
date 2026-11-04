"""Run the whole data stage: fetch -> build -> verify -> benchmarks.

Run:  python -m src.data.pipeline            (downloads fresh data)
      python -m src.data.pipeline --offline  (rebuilds from the existing data/raw/ snapshot)
"""
import sys

from src.data import benchmarks, build, fetch, verify


def main() -> None:
    if "--offline" not in sys.argv:
        print("== fetch"); fetch.main()
    print("== build"); build.main()
    print("== benchmarks"); benchmarks.main()
    print("== verify"); verify.main()   # last, because it exits non-zero on FAIL


if __name__ == "__main__":
    main()
