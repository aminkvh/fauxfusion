// fauxfusion / ffusion
//
// Pure-visualization tool: takes a structure file and a chain ID, and
// generates a synthetic multi-MODEL PDB trajectory that goes from a random
// noise cloud to the chain's real coordinates -- mimicking the look of an
// RFdiffusion-style binder-generation animation.
//
// This is NOT a real diffusion model and does not reproduce any actual
// generative process (hence "faux" fusion). It just interpolates (with a
// DDPM-style noise schedule shape) between random per-atom noise and the
// true coordinates, so it *looks* like a structure condensing out of noise
// when played back frame by frame in PyMOL, VMD, ChimeraX, etc.
//
// No external crates -- Rust std only. Input may be PDB or mmCIF
// (auto-detected from the file extension); output is always multi-MODEL PDB.

use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

struct Atom {
    serial: i64,
    name: String,
    altloc: char,
    resname: String,
    chain: char,
    resseq: i64,
    icode: char,
    occupancy: f64,
    bfactor: f64,
    element: String,
    charge: String,
    het: bool,
    x: f64,
    y: f64,
    z: f64,
}

fn infer_element(name: &str) -> String {
    let letters: String = name.chars().filter(|c| c.is_alphabetic()).collect();
    if letters.is_empty() {
        "C".to_string()
    } else {
        letters[0..1].to_uppercase()
    }
}

fn safe_float(s: &str, default: f64) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        default
    } else {
        t.parse().unwrap_or(default)
    }
}

// Safe fixed-column substring: never panics even if the line is shorter
// than the requested range (real-world PDB files sometimes omit trailing
// optional columns like element/charge).
fn ss(line: &str, a: usize, b: usize) -> &str {
    let len = line.len();
    let a = a.min(len);
    let b = b.min(len).max(a);
    &line[a..b]
}

fn ch_at(line: &str, i: usize) -> char {
    line.as_bytes().get(i).map(|&b| b as char).unwrap_or(' ')
}

type Chains = HashMap<char, Vec<Atom>>;

// ------------------------------------------------------------------ PDB in

fn parse_pdb(path: &str) -> io::Result<(Vec<char>, Chains)> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut chain_order: Vec<char> = Vec::new();
    let mut chains: Chains = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        if line.len() < 6 {
            continue;
        }
        let rec = &line[0..6];
        if rec != "ATOM  " && rec != "HETATM" {
            continue;
        }
        let name_raw = ss(&line, 12, 16);
        let element_field = ss(&line, 76, 78).trim().to_string();
        let element = if element_field.is_empty() { infer_element(name_raw) } else { element_field };
        let chain = ch_at(&line, 21);
        let charge_raw = ss(&line, 78, 80).to_string();

        let atom = Atom {
            serial: ss(&line, 6, 11).trim().parse().unwrap_or(0),
            name: name_raw.trim().to_string(),
            altloc: ch_at(&line, 16),
            resname: ss(&line, 17, 20).trim().to_string(),
            chain,
            resseq: ss(&line, 22, 26).trim().parse().unwrap_or(0),
            icode: ch_at(&line, 26),
            occupancy: safe_float(ss(&line, 54, 60), 1.0),
            bfactor: safe_float(ss(&line, 60, 66), 0.0),
            element,
            charge: charge_raw,
            het: rec == "HETATM",
            x: ss(&line, 30, 38).trim().parse().unwrap_or(0.0),
            y: ss(&line, 38, 46).trim().parse().unwrap_or(0.0),
            z: ss(&line, 46, 54).trim().parse().unwrap_or(0.0),
        };

        chains.entry(chain).or_insert_with(|| {
            chain_order.push(chain);
            Vec::new()
        }).push(atom);
    }
    Ok((chain_order, chains))
}

// ---------------------------------------------------------------- mmCIF in

fn tokenize_cif_row(line: &str) -> Vec<String> {
    let chars: Vec<char> = line.chars().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        while i < n && chars[i].is_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        if chars[i] == '\'' || chars[i] == '"' {
            let quote = chars[i];
            i += 1;
            let start = i;
            while i < n && chars[i] != quote {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            i += 1;
        } else {
            let start = i;
            while i < n && !chars[i].is_whitespace() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
        }
    }
    tokens
}

fn cif_field<'a>(tokens: &'a [String], col_index: &HashMap<String, usize>, name: &str, default: &'a str) -> &'a str {
    match col_index.get(name) {
        Some(&idx) if idx < tokens.len() => tokens[idx].as_str(),
        _ => default,
    }
}

fn resolve(v: &str, missing: bool) -> String {
    if missing || v == "?" || v == "." {
        String::new()
    } else {
        v.to_string()
    }
}

/// Minimal mmCIF reader: only extracts the _atom_site loop, first model.
fn parse_cif(path: &str) -> io::Result<(Vec<char>, Chains)> {
    let content = fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();

    let mut chain_order: Vec<char> = Vec::new();
    let mut chains: Chains = HashMap::new();

    let mut i = 0;
    while i < n {
        if lines[i].trim() != "loop_" {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        let mut columns: Vec<String> = Vec::new();
        while j < n && lines[j].trim().starts_with("_atom_site.") {
            columns.push(lines[j].trim()["_atom_site.".len()..].to_string());
            j += 1;
        }
        if columns.is_empty() {
            i += 1;
            continue;
        }
        let mut col_index: HashMap<String, usize> = HashMap::new();
        for (idx, name) in columns.iter().enumerate() {
            col_index.insert(name.clone(), idx);
        }

        let mut first_model: Option<String> = None;
        let mut k = j;
        while k < n {
            let stripped = lines[k].trim();
            if stripped.is_empty() {
                k += 1;
                continue;
            }
            if !(stripped.starts_with("ATOM") || stripped.starts_with("HETATM")) {
                break;
            }
            let tokens = tokenize_cif_row(stripped);
            if tokens.len() < columns.len() {
                k += 1;
                continue;
            }

            let model_num = cif_field(&tokens, &col_index, "pdbx_PDB_model_num", "1").to_string();
            if first_model.is_none() {
                first_model = Some(model_num.clone());
            }
            if first_model.as_deref() != Some(model_num.as_str()) {
                k += 1;
                continue;
            }

            let mut chain_s = cif_field(&tokens, &col_index, "auth_asym_id", "?").to_string();
            if chain_s == "?" || chain_s == "." {
                chain_s = cif_field(&tokens, &col_index, "label_asym_id", " ").to_string();
            }
            let mut name_s = cif_field(&tokens, &col_index, "auth_atom_id", "?").to_string();
            if name_s == "?" || name_s == "." {
                name_s = cif_field(&tokens, &col_index, "label_atom_id", "").to_string();
            }
            let mut resname_s = cif_field(&tokens, &col_index, "auth_comp_id", "?").to_string();
            if resname_s == "?" || resname_s == "." {
                resname_s = cif_field(&tokens, &col_index, "label_comp_id", "").to_string();
            }
            let resseq_s = cif_field(&tokens, &col_index, "auth_seq_id", "0").to_string();
            let icode_s = cif_field(&tokens, &col_index, "pdbx_PDB_ins_code", "?").to_string();
            let element_s = cif_field(&tokens, &col_index, "type_symbol", "").to_string();
            let het = cif_field(&tokens, &col_index, "group_PDB", "ATOM") == "HETATM";
            let serial_s = cif_field(&tokens, &col_index, "id", "0").to_string();
            let occ_s = cif_field(&tokens, &col_index, "occupancy", "1.0").to_string();
            let bfac_s = cif_field(&tokens, &col_index, "B_iso_or_equiv", "0.0").to_string();

            let chain_char = chain_s.chars().next().unwrap_or(' ');
            let name_resolved = resolve(&name_s, false);
            let element_resolved = resolve(&element_s, false);
            let element_final = if element_resolved.is_empty() { infer_element(&name_resolved) } else { element_resolved };
            let icode_resolved = resolve(&icode_s, false);

            let atom = Atom {
                serial: if serial_s != "?" && serial_s != "." { serial_s.parse().unwrap_or(0) } else { 0 },
                name: name_resolved,
                altloc: ' ',
                resname: resolve(&resname_s, false),
                chain: chain_char,
                resseq: if resseq_s != "?" && resseq_s != "." { resseq_s.parse().unwrap_or(0) } else { 0 },
                icode: icode_resolved.chars().next().unwrap_or(' '),
                occupancy: if occ_s != "?" && occ_s != "." { safe_float(&occ_s, 1.0) } else { 1.0 },
                bfactor: if bfac_s != "?" && bfac_s != "." { safe_float(&bfac_s, 0.0) } else { 0.0 },
                element: element_final,
                charge: "  ".to_string(),
                het,
                x: cif_field(&tokens, &col_index, "Cartn_x", "0").parse().unwrap_or(0.0),
                y: cif_field(&tokens, &col_index, "Cartn_y", "0").parse().unwrap_or(0.0),
                z: cif_field(&tokens, &col_index, "Cartn_z", "0").parse().unwrap_or(0.0),
            };

            chains.entry(chain_char).or_insert_with(|| {
                chain_order.push(chain_char);
                Vec::new()
            }).push(atom);

            k += 1;
        }
        i = k;
    }
    Ok((chain_order, chains))
}

fn load_structure(path: &str) -> io::Result<(Vec<char>, Chains)> {
    let lower = path.to_lowercase();
    if lower.ends_with(".cif") || lower.ends_with(".mmcif") {
        parse_cif(path)
    } else {
        parse_pdb(path)
    }
}

// ----------------------------------------------------------------- PDB out

fn format_atom_name(name: &str) -> String {
    let name = name.trim();
    if name.chars().count() >= 4 {
        name.chars().take(4).collect()
    } else {
        format!(" {:<3}", name)
    }
}

fn format_pdb_atom_line(atom: &Atom, x: f64, y: f64, z: f64) -> String {
    let record = if atom.het { "HETATM" } else { "ATOM" };
    let name_field = format_atom_name(&atom.name);
    let altloc = if atom.altloc == '\0' { ' ' } else { atom.altloc };
    let resname3: String = {
        let truncated: String = atom.resname.chars().take(3).collect();
        format!("{:>3}", truncated)
    };
    let icode = if atom.icode == '\0' { ' ' } else { atom.icode };
    let element2: String = {
        let truncated: String = atom.element.chars().take(2).collect();
        format!("{:>2}", truncated)
    };
    let charge2 = format!("{:>2}", atom.charge.trim());

    format!(
        "{:<6}{:>5} {}{}{} {}{:>4}{}   {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}          {}{}",
        record,
        atom.serial.rem_euclid(100000),
        name_field,
        altloc,
        resname3,
        atom.chain,
        atom.resseq.rem_euclid(10000),
        icode,
        x, y, z,
        atom.occupancy, atom.bfactor,
        element2,
        charge2,
    )
}

// ---------------------------------------------------------------- schedule

fn alpha_bar(progress: f64, schedule: &str) -> f64 {
    if schedule == "linear" {
        progress
    } else {
        (progress * std::f64::consts::FRAC_PI_2).sin().powi(2) // cosine: eased start/end
    }
}

// -------------------------------------------------------------------- RNG

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }
    fn gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn default_seed() -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (process::id() as u64).wrapping_mul(0x2545_F491_4F6C_DD1D)
}

// --------------------------------------------------------------------- CLI

enum ChainSel {
    All,
    Some(Vec<char>),
}

/// Parses "A", "A,B,C", or "all" (case-insensitive) into a ChainSel.
fn parse_chain_sel(raw: &str) -> Result<ChainSel, String> {
    if raw.eq_ignore_ascii_case("all") {
        return Ok(ChainSel::All);
    }
    let mut chains = Vec::new();
    for tok in raw.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let mut it = tok.chars();
        let c = it.next().unwrap();
        if it.next().is_some() {
            return Err(format!("invalid chain id '{}': chain ids are a single character", tok));
        }
        chains.push(c);
    }
    if chains.is_empty() {
        return Err("--chain/-c requires at least one chain id, a comma-separated list, or 'all'".to_string());
    }
    Ok(ChainSel::Some(chains))
}

struct Args {
    structure: String,
    chain_sel: ChainSel,
    output: Option<String>,
    frames: i64,
    schedule: String,
    noise_scale: f64,
    seed: Option<u64>,
    no_context: bool,
}

fn print_help() {
    println!("usage: ffusion <structure> --chain CHAIN[,CHAIN...]|all [-o OUTPUT] [--frames N]");
    println!("               [--schedule cosine|linear] [--noise-scale F] [--seed N] [--no-context]");
    println!();
    println!("fauxfusion: generate a fake 'diffusion' trajectory (noise -> final structure)");
    println!("for one or more chains of a structure file, for visualization only.");
    println!();
    println!("positional arguments:");
    println!("  structure           input structure file, .pdb or .cif/.mmcif");
    println!();
    println!("options:");
    println!("  -h, --help          show this help message and exit");
    println!("  --chain, -c SEL     chain(s) to animate: single id, comma-separated list,");
    println!("                      or 'all' (required)");
    println!("  -o, --output PATH   output trajectory PDB (default: <input>_traj.pdb)");
    println!("  --frames, -n N      number of frames (default 60)");
    println!("  --schedule S        cosine (default, eased start/end) or linear");
    println!("  --noise-scale F     starting noise-cloud radius, x Rg of each animated chain (default 1.2)");
    println!("  --seed N            RNG seed for reproducibility");
    println!("  --no-context        drop non-animated chains; output only the animated chain(s)");
}

fn parse_args() -> Args {
    let argv: Vec<String> = env::args().skip(1).collect();
    let mut structure: Option<String> = None;
    let mut chain_raw: Option<String> = None;
    let mut output: Option<String> = None;
    let mut frames: i64 = 60;
    let mut schedule = "cosine".to_string();
    let mut noise_scale: f64 = 1.2;
    let mut seed: Option<u64> = None;
    let mut no_context = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            }
            "--chain" | "-c" => {
                i += 1;
                chain_raw = argv.get(i).cloned();
            }
            "-o" | "--output" => {
                i += 1;
                output = argv.get(i).cloned();
            }
            "--frames" | "-n" => {
                i += 1;
                frames = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(60);
            }
            "--schedule" => {
                i += 1;
                schedule = argv.get(i).cloned().unwrap_or_else(|| "cosine".to_string());
            }
            "--noise-scale" => {
                i += 1;
                noise_scale = argv.get(i).and_then(|s| s.parse().ok()).unwrap_or(1.2);
            }
            "--seed" => {
                i += 1;
                seed = argv.get(i).and_then(|s| s.parse().ok());
            }
            "--no-context" => no_context = true,
            other => {
                if structure.is_none() && !other.starts_with('-') {
                    structure = Some(other.to_string());
                } else {
                    eprintln!("ffusion: unrecognized argument '{}'", other);
                    process::exit(2);
                }
            }
        }
        i += 1;
    }

    let structure = structure.unwrap_or_else(|| {
        eprintln!("ffusion: missing required argument: structure");
        process::exit(2);
    });
    let chain_raw = chain_raw.unwrap_or_else(|| {
        eprintln!("ffusion: missing required argument: --chain/-c");
        process::exit(2);
    });
    let chain_sel = parse_chain_sel(&chain_raw).unwrap_or_else(|e| {
        eprintln!("ffusion: {}", e);
        process::exit(2);
    });
    if schedule != "cosine" && schedule != "linear" {
        eprintln!("ffusion: --schedule must be 'cosine' or 'linear'");
        process::exit(2);
    }

    Args { structure, chain_sel, output, frames, schedule, noise_scale, seed, no_context }
}

// -------------------------------------------------------------------- main

fn main() {
    let args = parse_args();

    let out_path = args.output.clone().unwrap_or_else(|| match args.structure.rfind('.') {
        Some(idx) => format!("{}_traj.pdb", &args.structure[..idx]),
        None => format!("{}_traj.pdb", args.structure),
    });

    let (chain_order, chains) = match load_structure(&args.structure) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("ffusion: could not read '{}': {}", args.structure, e);
            process::exit(1);
        }
    };

    let requested: Vec<char> = match &args.chain_sel {
        ChainSel::All => chain_order.clone(),
        ChainSel::Some(list) => list.clone(),
    };
    let missing: Vec<char> = requested
        .iter()
        .filter(|c| chains.get(c).is_none_or(|a| a.is_empty()))
        .cloned()
        .collect();
    if !missing.is_empty() {
        let avail: Vec<String> = chain_order.iter().map(|c| c.to_string()).collect();
        let missing_s: Vec<String> = missing.iter().map(|c| c.to_string()).collect();
        eprintln!("Chain(s) {} not found. Available chains: {}", missing_s.join(", "), avail.join(", "));
        process::exit(1);
    }
    // dedupe, preserving file order (matters for --chain all and RNG draw order)
    let animate: HashSet<char> = requested.into_iter().collect();
    let animate_order: Vec<char> = chain_order.iter().filter(|c| animate.contains(c)).cloned().collect();

    let seed = args.seed.unwrap_or_else(default_seed);
    let mut rng = SplitMix64::new(seed);

    struct ChainMotion {
        cx: f64,
        cy: f64,
        cz: f64,
        x0_dev: Vec<(f64, f64, f64)>,
        noise_dev: Vec<(f64, f64, f64)>,
        rg: f64,
    }

    let mut motions: HashMap<char, ChainMotion> = HashMap::new();
    for &cid in &animate_order {
        let atoms = &chains[&cid];
        let n_atoms_f = atoms.len() as f64;
        let cx = atoms.iter().map(|a| a.x).sum::<f64>() / n_atoms_f;
        let cy = atoms.iter().map(|a| a.y).sum::<f64>() / n_atoms_f;
        let cz = atoms.iter().map(|a| a.z).sum::<f64>() / n_atoms_f;

        let x0_dev: Vec<(f64, f64, f64)> = atoms.iter().map(|a| (a.x - cx, a.y - cy, a.z - cz)).collect();
        let rg = (x0_dev.iter().map(|(dx, dy, dz)| dx * dx + dy * dy + dz * dz).sum::<f64>() / n_atoms_f).sqrt();
        let spread = rg * args.noise_scale;

        let raw_noise: Vec<(f64, f64, f64)> = (0..atoms.len())
            .map(|_| (rng.gauss() * spread, rng.gauss() * spread, rng.gauss() * spread))
            .collect();
        let nmx = raw_noise.iter().map(|v| v.0).sum::<f64>() / n_atoms_f;
        let nmy = raw_noise.iter().map(|v| v.1).sum::<f64>() / n_atoms_f;
        let nmz = raw_noise.iter().map(|v| v.2).sum::<f64>() / n_atoms_f;
        // zero-mean noise: each animated chain's own centroid never drifts frame to frame
        let noise_dev: Vec<(f64, f64, f64)> = raw_noise.iter().map(|(x, y, z)| (x - nmx, y - nmy, z - nmz)).collect();

        motions.insert(cid, ChainMotion { cx, cy, cz, x0_dev, noise_dev, rg });
    }

    let chains_to_keep: Vec<char> = if args.no_context { animate_order.clone() } else { chain_order.clone() };

    let file = match File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ffusion: could not write '{}': {}", out_path, e);
            process::exit(1);
        }
    };
    let mut w = BufWriter::new(file);
    let frames_n = args.frames.max(1);

    for frame_idx in 0..frames_n {
        // Last frame is always the untouched input, never the interpolation
        // formula -- guarantees an exact match even if a future schedule
        // doesn't land alpha_bar exactly on 1.0.
        let is_last = frame_idx == frames_n - 1;
        let progress = if frames_n > 1 { frame_idx as f64 / (frames_n - 1) as f64 } else { 1.0 };
        let a_bar = alpha_bar(progress, &args.schedule);
        let sa = a_bar.sqrt();
        let sn = (1.0 - a_bar).sqrt();

        writeln!(w, "MODEL     {:>4}", frame_idx + 1).unwrap();
        for &chain_id in &chains_to_keep {
            let atoms = &chains[&chain_id];
            let motion = motions.get(&chain_id);
            for (i, atom) in atoms.iter().enumerate() {
                let (x, y, z) = match motion {
                    Some(m) if !is_last => {
                        let (dx, dy, dz) = m.x0_dev[i];
                        let (nx, ny, nz) = m.noise_dev[i];
                        (m.cx + sa * dx + sn * nx, m.cy + sa * dy + sn * ny, m.cz + sa * dz + sn * nz)
                    }
                    _ => (atom.x, atom.y, atom.z),
                };
                writeln!(w, "{}", format_pdb_atom_line(atom, x, y, z)).unwrap();
            }
            writeln!(w, "TER").unwrap();
        }
        writeln!(w, "ENDMDL").unwrap();
    }
    writeln!(w, "END").unwrap();
    w.flush().unwrap();

    println!("Wrote {} frames to {}", frames_n, out_path);
    for &cid in &animate_order {
        let m = &motions[&cid];
        println!(
            "Animated chain '{}' ({} atoms), rg={:.2} A, noise spread={:.2} A",
            cid,
            chains[&cid].len(),
            m.rg,
            m.rg * args.noise_scale
        );
    }
    if !args.no_context {
        let others: Vec<String> = chains_to_keep.iter().filter(|c| !animate.contains(c)).map(|c| c.to_string()).collect();
        if !others.is_empty() {
            println!("Kept static context chains: {}", others.join(", "));
        }
    }
}
