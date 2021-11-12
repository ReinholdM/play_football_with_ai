from argparse import ArgumentParser
import os
import pathlib


parser = ArgumentParser()
parser.add_argument("--dump_dir", type=str)
parser.add_argument("--render_mode", type=str, default="reply")

if __name__ == "__main__":
    args = parser.parse_args()

    base_root = pathlib.Path(__file__).parent.parent.parent.absolute()
    print(f"sript working path: {base_root}")

    cmd_path = base_root / "dependencies/football/gfootball"
    cmd = cmd_path / (f"{args.render_mode}.py")
    print(cmd)

    for dump_file_path in pathlib.Path(args.dump_dir).glob("**/*.dump"):
        os.system(f"python {cmd} --trace_file {dump_file_path}")
