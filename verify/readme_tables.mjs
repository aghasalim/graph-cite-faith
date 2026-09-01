// Every number in the README's three data tables, recomputed from the records.
//
// The tables in README.md were pasted there by hand from the tables printed by
// experiments/counterfactual.py. Nothing has ever checked that the paste is
// still faithful, and a stale README is the failure that survives longest:
// reports/ can be regenerated and the prose around it left saying the old
// thing. The three tables here quote 34 numbers between them and two of the
// three quote numbers that appear in no CSV at all -- the share of replies
// answering "neither" is computed nowhere but in section 2.3.
//
// This reads README.md, finds each table by its header row, and requires every
// cell to match a value derived from reports/counterfactual.json. Node because
// the pairing for label sensitivity and the parsing of markdown are both
// natural here, and because a fourth Wilson interval written against the same
// formula from a different keyboard is the point of this directory.
//
// Run: node verify/readme_tables.mjs <root>

import { readFileSync } from "node:fs";
import { join } from "node:path";

const root = process.argv[2] ?? ".";
const Z = 1.96;

let failures = 0;
const fail = (msg) => {
  console.log(`  FAIL ${msg}`);
  failures += 1;
};

// The Wilson score interval as the published CSVs print it. Wilson rather than
// the normal approximation: two of these cells sit at exactly 0.000.
function wilson(k, n) {
  const p = k / n;
  const z2 = Z * Z;
  const den = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / den;
  const half = (Z * Math.sqrt((p * (1 - p)) / n + z2 / (4 * n * n))) / den;
  const lo = Math.max(0, centre - half);
  const hi = Math.min(1, centre + half);
  return `${p.toFixed(3)} [${lo.toFixed(3)},${hi.toFixed(3)}]`;
}

const records = JSON.parse(
  readFileSync(join(root, "reports", "counterfactual.json"), "utf8"),
);
// Every table below is the GNNExplainer arm. The saliency arm ran for one
// model, so pooling the two would weight that model twice.
const arm = records.filter((r) => r.explainer === "gnnexplainer");
if (arm.length === 0) throw new Error("no gnnexplainer records");

const models = [...new Set(arm.map((r) => r.model))].sort();

// The README names models without their provider prefix or their serving
// suffix: "llama-3.1-8b" for "llama-3.1-8b-instant". Resolved by requiring
// exactly one full id whose bare name starts with the short one, so a new model
// that made the short name ambiguous would be an error rather than a silent
// pick of the first match.
function resolve(short) {
  const hit = models.filter((m) => m.split("/").pop().startsWith(short));
  if (hit.length !== 1) {
    fail(`"${short}" matches ${hit.length} models, not 1`);
    return null;
  }
  return hit[0];
}

const share = (rows, pred) => [rows.filter(pred).length, rows.length];

const stats = new Map();
for (const model of models) {
  const mine = arm.filter((r) => r.model === model);
  const control = mine.filter((r) => r.kind === "control");
  const narrate = mine.filter((r) => r.kind === "narrate");
  // The decisive cell for the naive measure: a decoy subgraph carrying the
  // node's real label, so structure and label point at different answers.
  const decisive = narrate.filter(
    (r) => r.subgraph === "decoy" && r.label === "true",
  );

  // Label sensitivity: same node, same subgraph, temperature 0, the two
  // prompts differ in one word. Does the named motif move?
  const pairs = new Map();
  for (const r of narrate) {
    const key = `${r.node}|${r.subgraph}`;
    if (!pairs.has(key)) pairs.set(key, {});
    pairs.get(key)[r.label] = r.motif_claimed;
  }
  const complete = [...pairs.values()].filter(
    (p) => p.true !== undefined && p.flipped !== undefined,
  );

  // "neither" in the four narration cells, as a range, which is how section
  // 2.3 quotes it.
  const cellShares = [];
  for (const subgraph of ["true", "decoy"]) {
    for (const label of ["true", "flipped"]) {
      const cell = narrate.filter(
        (r) => r.subgraph === subgraph && r.label === label,
      );
      cellShares.push(
        cell.filter((r) => r.motif_claimed === "neither").length / cell.length,
      );
    }
  }

  const [ctlK, ctlN] = share(control, (r) => r.agrees_with_structure);
  const [labK, labN] = share(decisive, (r) => r.agrees_with_label === true);
  stats.set(model, {
    edgeReading: ctlK / ctlN,
    edgeReadingCI: wilson(ctlK, ctlN),
    controlN: ctlN,
    followsLabel: labK / labN,
    sensMoved: complete.filter((p) => p.true !== p.flipped).length,
    sensN: complete.length,
    neitherControl:
      control.filter((r) => r.motif_claimed === "neither").length / ctlN,
    neitherLo: Math.min(...cellShares),
    neitherHi: Math.max(...cellShares),
  });
}

// A markdown table located by its header row and read to the first blank line.
// Locating by header means a table moving in the document does not break this,
// and a table being deleted does.
const readme = readFileSync(join(root, "README.md"), "utf8").split("\n");
function table(headerStartsWith) {
  const at = readme.findIndex((l) => l.startsWith(headerStartsWith));
  if (at < 0) {
    fail(`no table in README.md with header "${headerStartsWith}"`);
    return [];
  }
  const rows = [];
  for (let i = at + 2; i < readme.length && readme[i].startsWith("|"); i++) {
    rows.push(
      readme[i]
        .replace(/^\||\|$/g, "")
        .split("|")
        .map((c) => c.trim().replace(/\*\*/g, "")),
    );
  }
  return rows;
}

function cmp(what, got, want) {
  if (String(got) === String(want)) return true;
  fail(`${what}: README says ${want}, the records give ${got}`);
  return false;
}

let checked = 0;
const f3 = (x) => x.toFixed(3);

console.log("README table: the edge-reading control");
for (const [short, n, ci] of table("| model | n | edge-reading accuracy |")) {
  const model = resolve(short);
  if (!model) continue;
  const s = stats.get(model);
  const ok =
    [cmp(`${short} n`, s.controlN, n), cmp(`${short} accuracy`, s.edgeReadingCI, ci)]
      .every(Boolean);
  checked += 2;
  if (ok) console.log(`  ${short.padEnd(14)} n=${n}  ${ci}`);
}

console.log("\nREADME table: label sensitivity against edge reading");
for (const [short, er, naive, sens, npairs] of table(
  "| model | edge reading | label agr. (naive) |",
)) {
  const model = resolve(short);
  if (!model) continue;
  const s = stats.get(model);
  const ok = [
    cmp(`${short} edge reading`, f3(s.edgeReading), er),
    cmp(`${short} naive label agr.`, f3(s.followsLabel), naive),
    cmp(`${short} label sensitivity`, wilson(s.sensMoved, s.sensN), sens),
    cmp(`${short} n pairs`, s.sensN, npairs),
  ].every(Boolean);
  checked += 4;
  if (ok)
    console.log(
      `  ${short.padEnd(14)} read ${er}  naive ${naive}  sensitivity ${sens}  ${npairs} pairs`,
    );
}

console.log('\nREADME table: the share of replies answering "neither"');
for (const [short, ctl, present] of table(
  "| model | control (no label) | label present |",
)) {
  const model = resolve(short);
  if (!model) continue;
  const s = stats.get(model);
  // "0.190 to 0.333" across the four cells, or a single number when every cell
  // landed on the same share.
  const want = present.includes(" to ")
    ? present.split(" to ")
    : [present, present];
  const ok = [
    cmp(`${short} control neither`, f3(s.neitherControl), ctl),
    cmp(`${short} lowest cell`, f3(s.neitherLo), want[0]),
    cmp(`${short} highest cell`, f3(s.neitherHi), want[1]),
  ].every(Boolean);
  checked += 3;
  if (ok) console.log(`  ${short.padEnd(14)} control ${ctl}  narrated ${present}`);
}

if (checked !== 45) {
  fail(`checked ${checked} table cells, expected 45 across the three tables`);
}
if (failures > 0) {
  console.log(`\n${failures} of ${checked} README table cells disagree`);
  process.exit(1);
}
console.log(
  `\nJavaScript reproduces all ${checked} numbers in the three README tables ` +
    "from\nreports/counterfactual.json, including every Wilson interval",
);
