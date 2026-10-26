"""Show the geometry of a setup: interactive 3D view and x-z section.

Examples
--------
    python scripts/show_geometry.py                     # csi_747E
    python scripts/show_geometry.py csi_747E --cutaway 180
    python scripts/show_geometry.py csi_747E --save     # also write files
"""

import argparse

from albedo import load_setup, plotting
from albedo.visualize import section, view3d


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("setup", nargs="?", default="csi_747E",
                        help="setup name in configs/setups (default: csi_747E)")
    parser.add_argument("--cutaway", type=float, default=120.0,
                        help="angle of the wedge removed from the 3D view, in degrees")
    parser.add_argument("--save", action="store_true",
                        help="write figures/<setup>_3d.html and figures/<setup>_section.pdf/png")
    parser.add_argument("--no-section", action="store_true", help="skip the 2D section")
    args = parser.parse_args()

    setup = load_setup(args.setup)
    fig3d = view3d(setup, cutaway=args.cutaway)

    if args.save:
        plotting.figures_dir().mkdir(exist_ok=True)
        fig3d.write_html(plotting.figures_dir() / f"{setup.name}_3d.html")

    fig3d.show(renderer="browser")  # opens in the default web browser

    if not args.no_section:
        import matplotlib.pyplot as plt

        plotting.use_style()
        fig, _ = section(setup)
        if args.save:
            plotting.save(fig, f"{setup.name}_section")
        plt.show()


if __name__ == "__main__":
    main()
