# ABFE-IBS workflow

[中文](README.md) · [Documentation index](docs/README.md)

ABFE-IBS is an OpenMM-based absolute binding free-energy workflow using GROMACS
`.gro/.top` inputs and IBS/MBAR/TMBAR sampling. The engine is system-independent;
the Atenolol-rank11 files and `output*` directories in this workspace are a
reference system and must not be reused as checkpoints for another system.

## Start here

- [Installation, inputs, and running](docs/GETTING_STARTED.md)
- [Outputs, interpretation, and resume](docs/OUTPUTS_AND_RESUME.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Migrating to a new system](docs/MIGRATING_TO_A_NEW_SYSTEM.md)
- [Maintaining the code](docs/MAINTAINING.md)
- [Current audit status](docs/status/AUDIT_STATUS.md)

The detailed tutorials are currently maintained in Chinese. Run the CPU
pre-flight suite after code changes:

```bash
./tests/run_offline_tests.sh
```
