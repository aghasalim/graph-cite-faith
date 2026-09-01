// The scoring, re-derived from the record fields, and the identity section 2.2
// rests on.
//
// Every other program here recomputes an aggregate. This one goes underneath
// them, to the per-narration booleans they all add up. `agrees_with_label` was
// written by experiments/counterfactual.py from the node's true class, the
// class-name mapping and which of the two labels the prompt carried -- none of
// which survives into the published file. But it is recoverable a completely
// different way, from the shape of the subgraph that was actually shown plus
// the two condition strings, and if the two routes disagree the label
// agreement column is wrong and every label number in the README with it. Six
// of the seven instrument bugs in section 3 were bugs of exactly this shape.
//
// Then the claim the whole of section 2.2 turns on: a model that never answers
// "neither" has label agreement identically equal to 1 - structure agreement in
// the cells where structure and label point at different answers, so reading it
// as evidence of post-rationalisation is reading structure agreement backwards.
// That is an arithmetic identity, not an observation, and it can be checked.
//
// Run: java verify/Invariants.java <root>

import java.io.BufferedReader;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

public class Invariants {

    // Only the fields this program reasons about. The full 15-key structure of
    // every record is checked by verify/gocheck, which decodes the file with a
    // real JSON parser; the strict field count below means a record that lost
    // one of these five on the way here is an error rather than a default.
    record Rec(String model, String kind, String explainer, int node,
               String subgraph, String label, String shapeShown,
               String claimed, boolean parsed, boolean agreeStruct,
               Boolean agreeLabel) {}

    static int failures = 0;

    static void fail(String fmt, Object... args) {
        System.out.printf("  FAIL " + fmt + "%n", args);
        failures++;
    }

    static String other(String shape) {
        if (shape.equals("house")) return "cycle";
        if (shape.equals("cycle")) return "house";
        throw new IllegalArgumentException("not a motif shape: " + shape);
    }

    /** The motif the prompt's label points at, from the condition alone.
     *
     * With the true subgraph the node's own class is the shape shown, so the
     * true label points at that shape and the flipped label at the other one.
     * With a decoy the subgraph came from a node of the *other* class, so the
     * two swap. Nothing here consults the class labels the harness used.
     */
    static String labelMotif(String subgraph, String label, String shapeShown) {
        boolean pointsAtShown = subgraph.equals("true") == label.equals("true");
        return pointsAtShown ? shapeShown : other(shapeShown);
    }

    // The file is pandas' indent=2 layout, one key per line. A line is a field
    // only if it begins with exactly four spaces and the quoted key, so a key
    // name occurring inside the reply text is not mistaken for a field.
    static String strField(String line, String key) {
        String pat = "    \"" + key + "\":\"";
        if (!line.startsWith(pat)) return null;
        StringBuilder out = new StringBuilder();
        for (int i = pat.length(); i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') break;
            if (c == '\\' && i + 1 < line.length() && line.charAt(i + 1) == '/') continue;
            out.append(c);
        }
        return out.toString();
    }

    static String rawField(String line, String key) {
        String pat = "    \"" + key + "\":";
        if (!line.startsWith(pat)) return null;
        return line.substring(pat.length()).replace(",", "").trim();
    }

    static List<Rec> read(Path path) throws IOException {
        List<Rec> out = new ArrayList<>();
        Map<String, String> cur = new HashMap<>();
        int records = 0;
        try (BufferedReader r = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            while ((line = r.readLine()) != null) {
                if (line.startsWith("  {")) {
                    cur.clear();
                    continue;
                }
                if (line.startsWith("  }")) {
                    records++;
                    if (cur.size() != 11) {
                        throw new UncheckedIOException(new IOException(
                            "record " + records + " gave " + cur.size()
                            + " of the 11 fields this program needs; the file "
                            + "layout is not what it was written against"));
                    }
                    out.add(new Rec(cur.get("model"), cur.get("kind"),
                        cur.get("explainer"), Integer.parseInt(cur.get("node")),
                        cur.get("subgraph"), cur.get("label"),
                        cur.get("shape_shown"), cur.get("motif_claimed"),
                        cur.get("parsed").equals("true"),
                        cur.get("agrees_with_structure").equals("true"),
                        cur.get("agrees_with_label").equals("null") ? null
                            : cur.get("agrees_with_label").equals("true")));
                    continue;
                }
                for (String k : new String[]{"model", "kind", "explainer",
                                             "subgraph", "label", "shape_shown",
                                             "motif_claimed"}) {
                    String v = strField(line, k);
                    if (v != null) cur.put(k, v);
                }
                for (String k : new String[]{"node", "parsed",
                                             "agrees_with_structure",
                                             "agrees_with_label"}) {
                    String v = rawField(line, k);
                    if (v != null) cur.put(k, v);
                }
            }
        }
        if (out.isEmpty()) throw new IOException("no records read from " + path);
        return out;
    }

    static final class Cell {
        int n, structure, label, neither, unparsed;
        boolean conflict;
    }

    public static void main(String[] args) throws IOException {
        Path root = Path.of(args.length > 0 ? args[0] : ".");
        List<Rec> recs = read(root.resolve("reports/counterfactual.json"));
        System.out.printf("read %d records%n", recs.size());

        // Every condition should have been collected exactly once. A duplicate
        // would inflate one cell of the 2x2 and nothing downstream reports the
        // denominator it used.
        System.out.println("\nevery condition appears exactly once");
        Set<String> seen = new HashSet<>();
        int dupes = 0;
        for (Rec r : recs) {
            String k = String.join("|", r.model(), r.kind(), r.explainer(),
                                   String.valueOf(r.node()), r.subgraph(), r.label());
            if (!seen.add(k)) {
                if (dupes < 5) fail("%s is in the file twice", k);
                dupes++;
            }
        }
        if (dupes == 0) {
            System.out.printf("  %d distinct conditions, no duplicates%n", seen.size());
        } else if (dupes >= 5) {
            fail("%d duplicated conditions in total", dupes);
        }

        // The scoring, re-derived.
        System.out.println("\nagrees_with_label, re-derived from the condition and the shape shown");
        int narrations = 0, mismatched = 0;
        for (Rec r : recs) {
            if (!r.kind().equals("narrate")) {
                if (r.agreeLabel() != null) {
                    fail("a control probe carries a label agreement");
                }
                continue;
            }
            narrations++;
            String want = labelMotif(r.subgraph(), r.label(), r.shapeShown());
            boolean derived = r.claimed().equals(want);
            if (r.agreeLabel() == null || derived != r.agreeLabel()) {
                if (mismatched < 5) {
                    fail("%s node %d %s/%s: claimed %s against a label pointing at %s, "
                         + "so agreement is %b, but the file says %s",
                         r.model(), r.node(), r.subgraph(), r.label(), r.claimed(),
                         want, derived, r.agreeLabel());
                }
                mismatched++;
            }
            // The structure column has the same property and is cheap here.
            if (r.agreeStruct() != r.claimed().equals(r.shapeShown())) {
                fail("%s node %d: agrees_with_structure disagrees with the claim",
                     r.model(), r.node());
            }
        }
        if (mismatched == 0) {
            System.out.printf("  all %d narrations agree with the re-derived scoring%n",
                              narrations);
        } else {
            fail("%d of %d narrations were scored differently", mismatched, narrations);
        }

        // The identity behind section 2.2.
        System.out.println("\nlabel agreement is 1 - structure agreement wherever no reply says \"neither\"");
        Map<String, Cell> cells = new TreeMap<>();
        for (Rec r : recs) {
            if (!r.kind().equals("narrate") || !r.explainer().equals("gnnexplainer")) continue;
            String key = r.model() + " " + r.subgraph() + "/" + r.label();
            Cell c = cells.computeIfAbsent(key, k -> new Cell());
            c.n++;
            if (r.agreeStruct()) c.structure++;
            if (Boolean.TRUE.equals(r.agreeLabel())) c.label++;
            if (r.claimed().equals("neither")) c.neither++;
            if (!r.parsed()) c.unparsed++;
            c.conflict = !labelMotif(r.subgraph(), r.label(), r.shapeShown())
                             .equals(r.shapeShown());
        }
        int clean = 0, held = 0;
        for (Map.Entry<String, Cell> e : cells.entrySet()) {
            Cell c = e.getValue();
            if (c.neither > 0 || c.unparsed > 0) {
                System.out.printf("  %-42s %d of %d replies are \"neither\" or unparsed, "
                                  + "the premise does not hold%n",
                                  e.getKey(), c.neither + c.unparsed, c.n);
                continue;
            }
            clean++;
            // In a conflict cell the two motifs are different answers, so every
            // reply agrees with exactly one of them. In an agreeing cell they
            // are the same answer, so the two counts must be equal.
            boolean ok = c.conflict ? c.structure + c.label == c.n
                                    : c.structure == c.label;
            if (ok) {
                held++;
                System.out.printf("  %-42s %s, structure %d + label %d = %d%n",
                                  e.getKey(), c.conflict ? "conflict" : "agreeing",
                                  c.structure, c.label, c.n);
            } else {
                fail("%s: structure %d, label %d, n %d, the identity fails",
                     e.getKey(), c.structure, c.label, c.n);
            }
        }
        if (clean == 0) {
            fail("no cell is free of \"neither\", so the identity was never tested");
        }
        System.out.printf("  the identity holds in %d of the %d cells where it applies, "
                          + "of %d%n", held, clean, cells.size());

        if (failures > 0) {
            System.out.printf("%n%d invariants failed%n", failures);
            System.exit(1);
        }
        System.out.printf("%nJava re-derives the scoring of all %d narrations, and confirms in "
                          + "every cell%nwhere it applies the identity section 2.2 rests on%n",
                          narrations);
    }
}
