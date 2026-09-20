# fauxfusion (ffusion)

Ran a de novo diffusion design and don't have the diffusion trajectory for
visualization? fauxfusion fakes one: starts from noise, eases into the real
coordinates of one or more chains. Not a real diffusion model, just
interpolation for visualization.

Zero dependencies, Rust std only.

## Usage

```
ffusion input.pdb --chain A -o traj.pdb
ffusion input.cif --chain B -o traj.pdb --frames 80 --seed 7 --no-context
ffusion input.pdb --chain A,C -o traj.pdb    # animate two chains at once
ffusion input.pdb --chain all -o traj.pdb    # animate every chain
```

Output is a multi-MODEL PDB trajectory. The last frame is always the
untouched input coordinates.

| Flag | Default | Meaning |
|---|---|---|
| `--chain`, `-c` | required | chain(s) to animate: single id, comma-separated list, or `all` |
| `-o`, `--output` | `<input>_traj.pdb` | output path |
| `--frames`, `-n` | 60 | frame count |
| `--schedule` | `cosine` | `cosine` or `linear` |
| `--noise-scale` | 1.2 | starting noise radius, x each animated chain's own Rg |
| `--seed` | random | RNG seed |
| `--no-context` | off | drop non-animated chains from output |

Input: PDB or mmCIF (by extension). Output is always PDB. Chains not
selected by `--chain` are kept as static context unless `--no-context`.

## Build

```
cargo build --release
```

Binary lands at `target/release/ffusion` (`.exe` on Windows).
`.github/workflows/build.yml` cross-builds Windows/Linux/macOS on a `v*` tag
push or manual trigger.

## License

MIT, see [LICENSE](LICENSE).
