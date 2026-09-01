//! Two things the Python could not afford, on the two pieces of maths this
//! repository publishes everywhere.
//!
//! 1. The Wilson interval appears 78 times in `reports/`, and every "the
//!    interval still contains 0.5" claim in the README is a claim about it.
//!    Both the Python in `experiments/counterfactual.py` and the checks in
//!    `verify/aggregate.sql` and `verify/wilson.c` use the same closed form, so
//!    they share any algebra error in it. This derives the interval a different
//!    way: by bisecting the score equation |p - p0| = z sqrt(p0(1-p0)/n) for
//!    its two roots, which is the definition the closed form is a solution of.
//!    Then it checks the two agree for every (k, n) with n up to 2000, roughly
//!    two million pairs, rather than only at the 54 points the CSVs publish.
//!
//! 2. The exact permutation p values in `verify/correlations.R` enumerate all
//!    120 relabellings. A Monte Carlo test with a million shuffles has to land
//!    on the same number, or one of the two is wrong about the test rather than
//!    about the arithmetic.

use std::env;
use std::fs;
use std::process::exit;

const Z: f64 = 1.96;
const EXHAUSTIVE_N: usize = 2000;
const ROOT_TOL: f64 = 1e-12;
/// At k = 0 and k = n the clamped bound lands a few ulps off the endpoint,
/// so an exact bracket test fails on arithmetic rather than on algebra. An
/// interval that genuinely failed to contain its own estimate would miss by
/// far more than this.
const BRACKET_TOL: f64 = 1e-9;
const SHUFFLES: usize = 1_000_000;
const SIGMA: f64 = 4.0;

/// xorshift64*. Not cryptographic and not meant to be: it needs to be uniform,
/// fast and seeded reproducibly, so a failure here can be re-run.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// The closed form the repository uses everywhere.
fn wilson_closed(k: u64, n: u64) -> (f64, f64) {
    let p = k as f64 / n as f64;
    let n = n as f64;
    let z2 = Z * Z;
    let den = 1.0 + z2 / n;
    let centre = (p + z2 / (2.0 * n)) / den;
    let half = Z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / den;
    ((centre - half).max(0.0), (centre + half).min(1.0))
}

/// The same interval as the two roots of the score equation, found by
/// bisection. Nothing here knows the algebraic solution.
fn wilson_roots(k: u64, n: u64) -> (f64, f64) {
    let p = k as f64 / n as f64;
    let nf = n as f64;
    // Positive outside the interval, negative inside it.
    let g = |p0: f64| (p - p0) * (p - p0) - Z * Z * p0 * (1.0 - p0) / nf;

    let bisect = |mut a: f64, mut b: f64| {
        // g(a) >= 0 and g(b) <= 0 on entry.
        for _ in 0..200 {
            let m = 0.5 * (a + b);
            if g(m) > 0.0 {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    };

    let lo = if k == 0 { 0.0 } else { bisect(0.0, p) };
    let hi = if k == n { 1.0 } else { bisect(1.0, p) };
    (lo, hi)
}

/// CSV field splitter that respects the quotes around the interval columns,
/// which contain a comma of their own.
fn fields(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for c in line.trim_end_matches(['\r', '\n']).chars() {
        match c {
            '"' => quoted = !quoted,
            ',' if !quoted => out.push(std::mem::take(&mut cur)),
            _ => cur.push(c),
        }
    }
    out.push(cur);
    out
}

fn column_of(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("no column {name}"))
}

/// "0.550 [0.452,0.644]" as its three numbers.
fn parse_interval(s: &str) -> Option<(f64, f64, f64)> {
    let (p, rest) = s.split_once(" [")?;
    let (lo, hi) = rest.strip_suffix(']')?.split_once(',')?;
    Some((p.parse().ok()?, lo.parse().ok()?, hi.parse().ok()?))
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for i in 0..x.len() {
        num += (x[i] - mx) * (y[i] - my);
        sx += (x[i] - mx) * (x[i] - mx);
        sy += (y[i] - my) * (y[i] - my);
    }
    num / (sx.sqrt() * sy.sqrt())
}

fn monte_carlo_p(x: &[f64], y: &[f64], seed: u64) -> f64 {
    let r0 = pearson(x, y).abs();
    let mut rng = Rng::new(seed);
    let mut hits = 0usize;
    let mut perm = y.to_vec();
    for _ in 0..SHUFFLES {
        for i in (1..perm.len()).rev() {
            perm.swap(i, rng.below(i + 1));
        }
        if pearson(x, &perm).abs() >= r0 - 1e-12 {
            hits += 1;
        }
    }
    hits as f64 / SHUFFLES as f64
}

fn check_published(root: &str, path: &str, columns: &[&str]) -> (usize, usize, f64) {
    let text = fs::read_to_string(format!("{root}/{path}"))
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut lines = text.lines();
    let header = fields(lines.next().expect("empty file"));
    let n_col = column_of(&header, "n");
    let cols: Vec<usize> = columns.iter().map(|c| column_of(&header, c)).collect();

    let (mut checked, mut bad, mut worst) = (0usize, 0usize, 0.0f64);
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f = fields(line);
        let n: u64 = f[n_col].parse().expect("n is not an integer");
        for &c in &cols {
            let (p, lo, hi) = match parse_interval(&f[c]) {
                Some(v) => v,
                None => {
                    println!("  FAIL {} is not an interval", f[c]);
                    bad += 1;
                    continue;
                }
            };
            // k is recoverable because the published proportion is k/n to three
            // decimals and n is at most 100 here.
            let k = (p * n as f64).round() as u64;
            let (rlo, rhi) = wilson_roots(k, n);
            let d = (rlo - lo).abs().max((rhi - hi).abs());
            if d > worst {
                worst = d;
            }
            // The published string is rounded to three decimals, so the root
            // finder has to round to the same string.
            let got = format!("{rlo:.3},{rhi:.3}");
            let want = format!("{lo:.3},{hi:.3}");
            if got != want {
                println!("  FAIL {}/{} k={k} n={n}: roots {got}, published {want}",
                         path, header[c], );
                bad += 1;
            }
            checked += 1;
        }
    }
    (checked, bad, worst)
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let mut failures = 0usize;

    println!("the published intervals, re-derived by bisecting the score equation");
    let (c1, b1, w1) = check_published(
        &root,
        "reports/counterfactual_summary.csv",
        &["structure_agreement", "label_agreement"],
    );
    let (c2, b2, w2) = check_published(
        &root,
        "reports/edge_reading_control.csv",
        &["edge_reading_accuracy"],
    );
    failures += b1 + b2;
    println!(
        "  {} intervals from the two CSVs, worst |d| against the published bounds {:.2e}",
        c1 + c2,
        w1.max(w2)
    );
    println!(
        "  citation validity is not checked here: its denominator is a count of\n  \
         cited node ids, which lives in counterfactual.json rather than the CSV,\n  \
         and verify/aggregate.sql and verify/wilson.c both recompute it from there"
    );

    // The published points are 54 of them. The closed form is used for every
    // proportion this repository will ever publish, so check it over the whole
    // domain instead.
    println!("\nclosed form against the roots, every (k, n) with n <= {EXHAUSTIVE_N}");
    let mut pairs = 0u64;
    let mut worst = 0.0f64;
    let mut worst_at = (0u64, 0u64);
    for n in 1..=EXHAUSTIVE_N as u64 {
        for k in 0..=n {
            let (clo, chi) = wilson_closed(k, n);
            let (rlo, rhi) = wilson_roots(k, n);
            let d = (clo - rlo).abs().max((chi - rhi).abs());
            if d > worst {
                worst = d;
                worst_at = (k, n);
            }
            let p = k as f64 / n as f64;
            if clo > p + BRACKET_TOL || chi < p - BRACKET_TOL || clo < 0.0 || chi > 1.0 {
                println!("  FAIL k={k} n={n}: [{clo},{chi}] does not bracket {p}");
                failures += 1;
            }
            pairs += 1;
        }
    }
    println!("  {pairs} pairs, worst |d| {worst:.2e} at k={} n={}", worst_at.0, worst_at.1);
    if worst > ROOT_TOL {
        println!("  FAIL the closed form is not the root of the score equation");
        failures += 1;
    }

    // The permutation p values, by sampling rather than by enumeration.
    println!("\nthe exact permutation p values, by {SHUFFLES} random shuffles");
    let text = fs::read_to_string(format!("{root}/reports/competence_vs_label.csv"))
        .expect("cannot read competence_vs_label.csv");
    let mut lines = text.lines();
    let header = fields(lines.next().expect("empty file"));
    let (c_expl, c_er, c_fl, c_ls) = (
        column_of(&header, "explainer"),
        column_of(&header, "edge_reading"),
        column_of(&header, "follows_label"),
        column_of(&header, "label_sensitivity"),
    );
    let (mut er, mut fl, mut ls) = (vec![], vec![], vec![]);
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f = fields(line);
        if f[c_expl] != "gnnexplainer" {
            continue;
        }
        er.push(f[c_er].parse::<f64>().unwrap());
        fl.push(f[c_fl].parse::<f64>().unwrap());
        ls.push(f[c_ls].parse::<f64>().unwrap());
    }
    if er.len() != 5 {
        println!("  FAIL expected 5 models, found {}", er.len());
        failures += 1;
    }
    // sd of a proportion from SHUFFLES draws, at the widest.
    let tol = SIGMA * (0.25 / SHUFFLES as f64).sqrt();
    for (name, y, exact) in [
        ("naive label agr.", &fl, 0.058_333_333_333_333_336),
        ("label sensitivity", &ls, 0.991_666_666_666_666_7),
    ] {
        let mc = monte_carlo_p(&er, y, 0x5EED_1234);
        let d = (mc - exact).abs();
        println!(
            "  {name:<18} r {:+.3}  MC p {mc:.5}  exact p {exact:.5}  |d| {d:.5}  {}",
            pearson(&er, y),
            if d <= tol { "ok" } else { "FAIL" }
        );
        if d > tol {
            failures += 1;
        }
    }
    println!("  tolerance {SIGMA:.0} sd of a {SHUFFLES} draw proportion, {tol:.5}");

    if failures > 0 {
        println!("\n{failures} checks failed");
        exit(1);
    }
    println!(
        "\nRust re-derives every published interval from the score equation and \
         lands\non both exact permutation p values"
    );
}
