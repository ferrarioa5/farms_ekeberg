# implicit_ekeberg

This repository was created for two purposes:
1. implicit Ekeberg muscle model with an arbitrary controller in MuJoCo via FARMS extensions
2. Use of the drag model and ekeberg musles without requiring farms_amphibious


Installation:

1. If FARMS (farms_core, farms_sim, farms_mujoco) is already installed in your system skip this point and go to point 2.
Otherwise, a tested version of farms was included and can be installed to reproduce the examples in this repositories.
To do this initialize the git submodules and running the setup script:

```bash
cd farms
git submodule update --init --recursive
python farms/setup_farms.py
```

2. Install the current package in editable mode:

```bash
cd ..
pip install -e .
```

Usage examples:






