import pickle
import sys

from tools.execution import _execute_user_code


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m tools.execution_worker <input.pkl> <output.pkl>")

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "rb") as f:
        payload = pickle.load(f)

    result, stdout, figure_png, error = _execute_user_code(payload["code"], payload["df"])

    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "result": result,
                "stdout": stdout,
                "figure_png": figure_png,
                "error": error,
            },
            f,
        )


if __name__ == "__main__":
    main()
