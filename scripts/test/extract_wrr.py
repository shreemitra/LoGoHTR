import typer

fpath = "Results_pred_symlg/Summary.txt"


def main(error_tol: int):
    struct_ln = None
    wrr_ln = None
    with open(fpath, "r") as f:
        for ln in f.readlines():
            if ln.startswith(" Structure"):
                struct_ln = ln
            if ln.startswith("Cum. Files"):
                wrr_ln = ln

    struct_rate = float(struct_ln.split()[1])
    correct_num = [int(x) for x in wrr_ln.split()[2 : 2 + error_tol]]
    total_num = int(wrr_ln.split()[-1])

    print(f"Struct Rate: {struct_rate}")
    for i, n in enumerate(correct_num):
        print(f"WRR {i} tolerated: {n / total_num * 100:.3f}")


if __name__ == "__main__":
    typer.run(main)
