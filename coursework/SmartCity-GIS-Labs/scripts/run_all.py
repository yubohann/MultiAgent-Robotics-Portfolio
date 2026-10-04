import sys
import exp1_arcscan
import exp2_rubbersheet
import exp3_topology
import exp4_siting
import exp5_gas


def main():
    exp1_arcscan.main()
    exp2_rubbersheet.main()
    exp3_topology.main()
    exp4_siting.main()
    exp5_gas.main()
    sys.stderr.write("run_all completed\n")


if __name__ == "__main__":
    main()
