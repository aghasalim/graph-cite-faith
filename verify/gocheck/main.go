// Structural validation of every tracked file under reports/, plus an
// independent recomputation of reports/competence_vs_label.csv.
//
// The three CSVs in reports/ are the evidence for every table in the README,
// and reports/counterfactual.json is the per-narration record all three are
// derived from. Nothing checked that any of them is well formed: a truncated
// write, a column that drifted, a NaN out of a division, or an interval whose
// bounds do not bracket its point estimate would all be invisible until someone
// read the table. This walks every file and then rebuilds the joint table from
// the records, which pandas also does in experiments/counterfactual.py.
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Rounded to three decimals in the published file, so agreement means the two
// agree to within half a unit in the last published place.
const tol = 5e-4

var interval = regexp.MustCompile(`^([01]\.\d{3}) \[([01]\.\d{3}),([01]\.\d{3})\]$`)

type record struct {
	Model       string `json:"model"`
	Kind        string `json:"kind"`
	Explainer   string `json:"explainer"`
	Node        int    `json:"node"`
	Subgraph    string `json:"subgraph"`
	Label       string `json:"label"`
	ShapeShown  string `json:"shape_shown"`
	Claimed     string `json:"motif_claimed"`
	Parsed      bool   `json:"parsed"`
	AgreeStruct bool   `json:"agrees_with_structure"`
	AgreeLabel  *bool  `json:"agrees_with_label"`
	CitedValid  int    `json:"cited_valid"`
	NCited      int    `json:"n_cited"`
}

var requiredKeys = []string{
	"model", "kind", "explainer", "node", "subgraph", "label", "shape_shown",
	"motif_edges_recovered", "motif_claimed", "parsed", "agrees_with_structure",
	"agrees_with_label", "cited_valid", "n_cited", "reply",
}

var answers = map[string]bool{"house": true, "cycle": true, "neither": true}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("only %d rows", len(rows))
	}
	return rows[0], rows[1:], nil
}

func col(header []string, name string) int {
	for i, h := range header {
		if h == name {
			return i
		}
	}
	return -1
}

// validateCSV reports every structural problem in one file rather than the
// first, so a broken run is diagnosed in one pass.
func validateCSV(path string) []string {
	var problems []string
	header, rows, err := readCSV(path)
	if err != nil {
		return []string{fmt.Sprintf("unreadable: %v", err)}
	}

	seen := map[string]bool{}
	for _, h := range header {
		if strings.TrimSpace(h) == "" {
			problems = append(problems, "a column has an empty name")
		}
		if seen[h] {
			problems = append(problems, fmt.Sprintf("duplicate column %q", h))
		}
		seen[h] = true
	}

	for i, row := range rows {
		for j, v := range row {
			if strings.TrimSpace(v) == "" {
				problems = append(problems,
					fmt.Sprintf("row %d: empty %s", i+2, header[j]))
				continue
			}
			// An interval column: check the string parses and that the bounds
			// actually bracket the point estimate. A published interval that
			// does not contain its own proportion is the failure mode the
			// README's "the interval still contains 0.5" claims would hide.
			if m := interval.FindStringSubmatch(v); m != nil {
				p, _ := strconv.ParseFloat(m[1], 64)
				lo, _ := strconv.ParseFloat(m[2], 64)
				hi, _ := strconv.ParseFloat(m[3], 64)
				if !(lo <= p && p <= hi) {
					problems = append(problems, fmt.Sprintf(
						"row %d: %s interval %s does not contain its estimate",
						i+2, header[j], v))
				}
				continue
			}
			if strings.Contains(v, "[") {
				problems = append(problems, fmt.Sprintf(
					"row %d: %s = %q is not a well formed interval", i+2, header[j], v))
				continue
			}
			if f, err := strconv.ParseFloat(v, 64); err == nil {
				if math.IsNaN(f) || math.IsInf(f, 0) {
					problems = append(problems,
						fmt.Sprintf("row %d: %s is %v", i+2, header[j], f))
				}
			}
		}
	}
	return problems
}

func loadRecords(path string) ([]record, []string, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	// Decoded twice on purpose: once as raw maps to prove every field is
	// present, once into the struct. Decoding straight into the struct turns a
	// missing column into a zero and a silently wrong count.
	var raw []map[string]json.RawMessage
	if err := json.Unmarshal(blob, &raw); err != nil {
		return nil, nil, err
	}
	var problems []string
	for i, m := range raw {
		if len(m) != len(requiredKeys) {
			problems = append(problems,
				fmt.Sprintf("record %d has %d fields, expected %d", i, len(m), len(requiredKeys)))
		}
		for _, k := range requiredKeys {
			if _, ok := m[k]; !ok {
				problems = append(problems, fmt.Sprintf("record %d has no %q", i, k))
			}
		}
	}
	var recs []record
	if err := json.Unmarshal(blob, &recs); err != nil {
		return nil, nil, err
	}
	for i, r := range recs {
		switch {
		case r.Kind != "narrate" && r.Kind != "control":
			problems = append(problems, fmt.Sprintf("record %d: kind %q", i, r.Kind))
		case r.Kind == "control" && r.AgreeLabel != nil:
			problems = append(problems,
				fmt.Sprintf("record %d: a control probe has a label agreement", i))
		case r.Kind == "narrate" && r.AgreeLabel == nil:
			problems = append(problems,
				fmt.Sprintf("record %d: a narration has no label agreement", i))
		}
		if r.Parsed != answers[r.Claimed] {
			problems = append(problems, fmt.Sprintf(
				"record %d: parsed=%v but motif_claimed=%q", i, r.Parsed, r.Claimed))
		}
		if r.AgreeStruct != (r.Claimed == r.ShapeShown) {
			problems = append(problems, fmt.Sprintf(
				"record %d: agrees_with_structure=%v but claimed %q against shown %q",
				i, r.AgreeStruct, r.Claimed, r.ShapeShown))
		}
		if r.CitedValid < 0 || r.CitedValid > r.NCited {
			problems = append(problems, fmt.Sprintf(
				"record %d: %d of %d cited ids valid", i, r.CitedValid, r.NCited))
		}
	}
	return recs, problems, nil
}

type arm struct{ model, explainer string }

// joint rebuilds one row of reports/competence_vs_label.csv from the records.
type joint struct {
	n                                      int
	edgeReading, followsStruct, followsLab float64
	sensitivity                            float64
	sensN                                  int
	unparsed                               float64
}

func recompute(recs []record) map[arm]joint {
	type counts struct{ ctlN, ctlOK, decN, decStruct, decLab, narN, narParsed int }
	c := map[arm]*counts{}
	// The within-node contrast: same node, same subgraph, temperature 0, the
	// prompts differ in one word. Keyed on everything but the label.
	pairs := map[[4]string]map[string]string{}

	get := func(a arm) *counts {
		if c[a] == nil {
			c[a] = &counts{}
		}
		return c[a]
	}
	for _, r := range recs {
		a := arm{r.Model, r.Explainer}
		k := get(a)
		if r.Kind == "control" {
			k.ctlN++
			if r.AgreeStruct {
				k.ctlOK++
			}
			continue
		}
		k.narN++
		if r.Parsed {
			k.narParsed++
		}
		key := [4]string{r.Model, r.Explainer, strconv.Itoa(r.Node), r.Subgraph}
		if pairs[key] == nil {
			pairs[key] = map[string]string{}
		}
		pairs[key][r.Label] = r.Claimed
		if r.Subgraph == "decoy" && r.Label == "true" {
			k.decN++
			if r.AgreeStruct {
				k.decStruct++
			}
			if r.AgreeLabel != nil && *r.AgreeLabel {
				k.decLab++
			}
		}
	}

	moved := map[arm][2]int{}
	for key, byLabel := range pairs {
		t, okT := byLabel["true"]
		f, okF := byLabel["flipped"]
		if !okT || !okF {
			continue
		}
		a := arm{key[0], key[1]}
		m := moved[a]
		m[0]++
		if t != f {
			m[1]++
		}
		moved[a] = m
	}

	out := map[arm]joint{}
	for a, k := range c {
		if k.ctlN == 0 || k.decN == 0 {
			continue
		}
		m := moved[a]
		out[a] = joint{
			n:             k.decN,
			edgeReading:   float64(k.ctlOK) / float64(k.ctlN),
			followsStruct: float64(k.decStruct) / float64(k.decN),
			followsLab:    float64(k.decLab) / float64(k.decN),
			sensitivity:   float64(m[1]) / float64(m[0]),
			sensN:         m[0],
			unparsed:      1 - float64(k.narParsed)/float64(k.narN),
		}
	}
	return out
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	failures := 0
	files := []string{
		"reports/counterfactual_summary.csv",
		"reports/edge_reading_control.csv",
		"reports/competence_vs_label.csv",
	}
	fmt.Println("structural validation")
	for _, rel := range files {
		problems := validateCSV(filepath.Join(*root, rel))
		if len(problems) == 0 {
			fmt.Printf("  %-42s ok\n", rel)
			continue
		}
		fmt.Printf("  %-42s %d problems\n", rel, len(problems))
		for _, p := range problems {
			fmt.Printf("      %s\n", p)
		}
		failures += len(problems)
	}

	recs, problems, err := loadRecords(filepath.Join(*root, "reports/counterfactual.json"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "counterfactual.json: %v\n", err)
		os.Exit(2)
	}
	if len(problems) == 0 {
		fmt.Printf("  %-42s ok, %d records\n", "reports/counterfactual.json", len(recs))
	} else {
		fmt.Printf("  %-42s %d problems\n", "reports/counterfactual.json", len(problems))
		for i, p := range problems {
			if i == 10 {
				fmt.Printf("      ... and %d more\n", len(problems)-10)
				break
			}
			fmt.Printf("      %s\n", p)
		}
		failures += len(problems)
	}

	// Every node in an arm contributes four narrations and one control probe,
	// which is what makes the 2x2 a 2x2. Partial nodes are dropped upstream, so
	// a mismatch here means the published file was written from a different set
	// of records than the one shipped.
	fmt.Println("\nthe 2x2 is balanced")
	byArm := map[arm]map[string]int{}
	for _, r := range recs {
		a := arm{r.Model, r.Explainer}
		if byArm[a] == nil {
			byArm[a] = map[string]int{}
		}
		byArm[a][r.Kind+"/"+r.Subgraph+"/"+r.Label]++
	}
	var arms []arm
	for a := range byArm {
		arms = append(arms, a)
	}
	sort.Slice(arms, func(i, j int) bool {
		if arms[i].model != arms[j].model {
			return arms[i].model < arms[j].model
		}
		return arms[i].explainer < arms[j].explainer
	})
	for _, a := range arms {
		cells := byArm[a]
		want := cells["control/true/none"]
		ok := want > 0
		for _, cell := range []string{"narrate/true/true", "narrate/true/flipped",
			"narrate/decoy/true", "narrate/decoy/flipped"} {
			if cells[cell] != want {
				ok = false
			}
		}
		if ok {
			fmt.Printf("  %-24s %-13s ok, %d nodes x (4 narrations + 1 control)\n",
				a.model, a.explainer, want)
		} else {
			fmt.Printf("  %-24s %-13s FAIL cells %v\n", a.model, a.explainer, cells)
			failures++
		}
	}

	fmt.Println("\nreports/competence_vs_label.csv, recomputed from the records")
	got := recompute(recs)
	header, rows, err := readCSV(filepath.Join(*root, "reports/competence_vs_label.csv"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "competence_vs_label.csv: %v\n", err)
		os.Exit(2)
	}
	numeric := []string{"edge_reading", "follows_structure", "follows_label",
		"label_sensitivity", "unparsed"}
	idx := map[string]int{}
	for _, name := range append([]string{"model", "explainer", "n", "sens_n"}, numeric...) {
		if idx[name] = col(header, name); idx[name] < 0 {
			fmt.Fprintf(os.Stderr, "no %s column in competence_vs_label.csv\n", name)
			os.Exit(2)
		}
	}
	worst := 0.0
	checked := 0
	for _, row := range rows {
		a := arm{row[idx["model"]], row[idx["explainer"]]}
		j, ok := got[a]
		if !ok {
			fmt.Printf("  %-24s %-13s FAIL no records for this arm\n", a.model, a.explainer)
			failures++
			continue
		}
		mine := map[string]float64{
			"edge_reading": j.edgeReading, "follows_structure": j.followsStruct,
			"follows_label": j.followsLab, "label_sensitivity": j.sensitivity,
			"unparsed": j.unparsed,
		}
		bad := 0
		for _, name := range numeric {
			want, err := strconv.ParseFloat(row[idx[name]], 64)
			if err != nil {
				fmt.Printf("  %-24s %-13s FAIL %s = %q\n", a.model, a.explainer, name, row[idx[name]])
				bad++
				continue
			}
			d := math.Abs(mine[name] - want)
			if d > worst {
				worst = d
			}
			if d > tol {
				fmt.Printf("  %-24s %-13s FAIL %s: Go %.6f, published %.3f\n",
					a.model, a.explainer, name, mine[name], want)
				bad++
			}
			checked++
		}
		for _, p := range []struct {
			name string
			mine int
		}{{"n", j.n}, {"sens_n", j.sensN}} {
			want, _ := strconv.Atoi(row[idx[p.name]])
			if want != p.mine {
				fmt.Printf("  %-24s %-13s FAIL %s: Go %d, published %d\n",
					a.model, a.explainer, p.name, p.mine, want)
				bad++
			}
			checked++
		}
		if bad == 0 {
			fmt.Printf("  %-24s %-13s ok, 7 values\n", a.model, a.explainer)
		}
		failures += bad
	}

	if failures > 0 {
		fmt.Printf("\n%d problems\n", failures)
		os.Exit(1)
	}
	fmt.Printf("\nGo validates 4 files and reproduces all %d values in "+
		"competence_vs_label.csv (worst |d| %.2e)\n", checked, worst)
}
