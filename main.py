import argparse

from utils import Master

# group1 = parser.add_argument_group('group1', 'group1 description')
# group1.add_argument('foo', help='foo help')
# group2 = parser.add_argument_group('group2', 'group2 description')
# group2.add_argument('--bar', help='bar help')

MODES = ("xyz", "feff", "chi")


def main():
    parser = argparse.ArgumentParser(
        'Create bunch of of files (*.xyz, feff *.inp, feff *.chi) using frames from GULP *.history file'
    )
    parser.add_argument("input", help="path to GULP *.history file")
    parser.add_argument("absorber", type=str, help="Chemical element tag from periodic table (e.g. Mn)")
    parser.add_argument(
        "--modes", nargs='+', default=[*MODES], help="Modes to be used during run. Default/allowed {0}".format(MODES)
    )
    parser.add_argument('-s', '--shift', action="store_true", help=(
        "Shift origin to the crystal structure box (default: True)"
    )
                        )

    parser.add_argument('-f', '--frames', type=int, default=1000, help=(
        "Number of frames to be used for JMOL input file creation. If 0, xyz files will be saved for all snapshots."
        + "Use negative value (simple -1) to disable this calculation. Default value is 1000"
    )
                        )
    parser.add_argument('-c', '--cores', type=int, default=1, help="No. of cores, used for calculation (default: 1)")
    parser.add_argument('--dry-run', action='store_true', help=(
        "Prevent executions. Input parameters print only. Hint: use it to check what values will be passed to the "
        "program")
                        )

    args = parser.parse_args()
    args.frames = None if args.frames < 0 else args.frames

    print("{:=^50}".format(" Arguments list "))
    for arg in vars(args):
        print("  -", arg, ":", getattr(args, arg))
    print("="*50)
    if args.dry_run:
        exit()
    for mode in args.modes:
        if mode not in MODES:
            print("Unknown mode was found: {}. It will be skipped during program run!".format(mode))
            args.modes.remove(mode)

    master = Master(
        input_file=args.input,
        absorber=args.absorber,
        cores=args.cores,
        shift=args.shift,
        frames=args.frames,
        modes=[m.lower() for m in args.modes],
    )
    master.execute()


if __name__ == '__main__':
    main()
