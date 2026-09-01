/* Recompute every published proportion and every Wilson interval, in C.
 *
 * Third independent implementation of the same numbers, after pandas in
 * experiments/counterfactual.py and SQL in verify/aggregate.sql. The interval
 * is the part worth checking twice: four of the twenty four cells sit at
 * exactly 0.000 or 1.000, the closed form is four lines of algebra with two
 * clamps in it, and every claim in the README that says "the interval still
 * contains 0.5" rests on it being right.
 *
 * Reads reports/counterfactual.json record by record, groups it exactly as the
 * pandas groupby does, and rebuilds the 78 interval strings printed in
 * reports/counterfactual_summary.csv and reports/edge_reading_control.csv.
 * Columns in both CSVs are resolved by name, so a column added upstream cannot
 * silently shift what this reads. Exits non-zero on the first disagreement.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 4096
#define MAX_CELLS 64
#define Z 1.96

typedef struct {
    char model[64], expl[32], sub[16], lab[16];
    long n, ks, kl, cv, ct;
} Cell;

static Cell cells[MAX_CELLS];
static int ncells;

static Cell *cell_for(const char *model, const char *expl,
                      const char *sub, const char *lab)
{
    for (int i = 0; i < ncells; i++)
        if (strcmp(cells[i].model, model) == 0 && strcmp(cells[i].expl, expl) == 0
            && strcmp(cells[i].sub, sub) == 0 && strcmp(cells[i].lab, lab) == 0)
            return &cells[i];
    if (ncells == MAX_CELLS) {
        fprintf(stderr, "more than %d cells\n", MAX_CELLS);
        exit(2);
    }
    Cell *c = &cells[ncells++];
    snprintf(c->model, sizeof c->model, "%s", model);
    snprintf(c->expl, sizeof c->expl, "%s", expl);
    snprintf(c->sub, sizeof c->sub, "%s", sub);
    snprintf(c->lab, sizeof c->lab, "%s", lab);
    c->n = c->ks = c->kl = c->cv = c->ct = 0;
    return c;
}

/* The Wilson score interval, formatted the way the published CSVs print it.
 * Wilson rather than the normal approximation because the latter runs outside
 * [0, 1] at the cells that sit on 0 and 1, which is where the README's
 * strongest claims live. */
static void wilson(long k, long n, char *out, size_t cap)
{
    const double p = (double)k / (double)n;
    const double z2 = Z * Z;
    const double den = 1.0 + z2 / (double)n;
    const double centre = (p + z2 / (2.0 * (double)n)) / den;
    const double half = Z * sqrt(p * (1.0 - p) / (double)n
                                 + z2 / (4.0 * (double)n * (double)n)) / den;
    double lo = centre - half, hi = centre + half;
    if (lo < 0.0) lo = 0.0;
    if (hi > 1.0) hi = 1.0;
    snprintf(out, cap, "%.3f [%.3f,%.3f]", p, lo, hi);
}

/* A JSON string value, with pandas' escaped forward slash put back. Only the
 * escapes that actually occur in these fields are handled: the model names
 * carry a slash and nothing else here is escaped. */
static int json_str(const char *line, const char *key, char *out, size_t cap)
{
    char pat[64];
    snprintf(pat, sizeof pat, "    \"%s\":\"", key);
    const size_t plen = strlen(pat);
    if (strncmp(line, pat, plen) != 0)
        return 0;
    const char *p = line + plen;
    size_t o = 0;
    while (*p && *p != '"' && o + 1 < cap) {
        if (p[0] == '\\' && p[1] == '/') p++;
        out[o++] = *p++;
    }
    out[o] = '\0';
    return 1;
}

static int json_scalar(const char *line, const char *key, const char **val)
{
    char pat[64];
    snprintf(pat, sizeof pat, "    \"%s\":", key);
    const size_t plen = strlen(pat);
    if (strncmp(line, pat, plen) != 0)
        return 0;
    *val = line + plen;
    return 1;
}

static long read_records(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }

    char line[LINE];
    char model[64] = "", expl[32] = "", kind[32] = "", sub[16] = "", lab[16] = "";
    long cited_valid = 0, n_cited = 0;
    int agr_struct = 0, agr_label = 0, seen = 0;
    long records = 0;
    const char *v;

    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "  {", 3) == 0) {
            seen = 0;
            continue;
        }
        if (strncmp(line, "  }", 3) == 0) {
            /* nine fields per record, every one required: a truncated or
             * reordered file must be an error rather than a silent zero */
            if (seen != 9) {
                fprintf(stderr, "record %ld has %d of 9 required fields\n",
                        records, seen);
                fclose(f);
                exit(2);
            }
            Cell *c = cell_for(model, expl,
                               strcmp(kind, "control") == 0 ? "" : sub,
                               strcmp(kind, "control") == 0 ? "" : lab);
            c->n++;
            c->ks += agr_struct;
            c->kl += agr_label;
            c->cv += cited_valid;
            c->ct += n_cited;
            records++;
            continue;
        }
        if (json_str(line, "model", model, sizeof model)) { seen++; continue; }
        if (json_str(line, "explainer", expl, sizeof expl)) { seen++; continue; }
        if (json_str(line, "kind", kind, sizeof kind)) { seen++; continue; }
        if (json_str(line, "subgraph", sub, sizeof sub)) { seen++; continue; }
        if (json_str(line, "label", lab, sizeof lab)) { seen++; continue; }
        if (json_scalar(line, "agrees_with_structure", &v)) {
            agr_struct = strncmp(v, "true", 4) == 0; seen++; continue;
        }
        if (json_scalar(line, "agrees_with_label", &v)) {
            agr_label = strncmp(v, "true", 4) == 0; seen++; continue;
        }
        if (json_scalar(line, "cited_valid", &v)) {
            cited_valid = atol(v); seen++; continue;
        }
        if (json_scalar(line, "n_cited", &v)) {
            n_cited = atol(v); seen++; continue;
        }
    }
    fclose(f);
    return records;
}

/* CSV field by index, quotes stripped. The interval columns contain a comma
 * inside their quotes, so a split on commas alone reads the wrong column. */
static const char *field(const char *line, int index)
{
    static char out[256];
    const char *p = line;
    int col = 0;
    while (1) {
        int quoted = (*p == '"');
        if (quoted) p++;
        const char *start = p;
        while (*p && *p != '\n' && !(quoted ? (*p == '"') : (*p == ',')))
            p++;
        if (col == index) {
            size_t n = (size_t)(p - start);
            if (n >= sizeof out) n = sizeof out - 1;
            memcpy(out, start, n);
            out[n] = '\0';
            return out;
        }
        if (quoted && *p == '"') p++;
        if (*p != ',') return NULL;
        p++;
        col++;
    }
}

static int column_of(const char *header, const char *name)
{
    for (int i = 0; ; i++) {
        const char *f = field(header, i);
        if (!f) return -1;
        if (strcmp(f, name) == 0) return i;
    }
}

static int check(const char *path, int is_control, int *checked)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }

    char header[LINE], line[LINE];
    if (!fgets(header, sizeof header, f)) { fclose(f); exit(2); }

    const int c_model = column_of(header, "model");
    const int c_expl = column_of(header, "explainer");
    const int c_n = column_of(header, "n");
    const int c_sub = is_control ? -1 : column_of(header, "subgraph");
    const int c_lab = is_control ? -1 : column_of(header, "label");
    if (c_model < 0 || c_expl < 0 || c_n < 0 || (!is_control && (c_sub < 0 || c_lab < 0))) {
        fprintf(stderr, "%s: a key column is missing\n", path);
        fclose(f);
        exit(2);
    }

    const char *names[3];
    int cols[3], nmetric;
    if (is_control) {
        names[0] = "edge_reading_accuracy";
        nmetric = 1;
    } else {
        names[0] = "structure_agreement";
        names[1] = "label_agreement";
        names[2] = "citation_validity";
        nmetric = 3;
    }
    for (int m = 0; m < nmetric; m++) {
        cols[m] = column_of(header, names[m]);
        if (cols[m] < 0) {
            fprintf(stderr, "%s: no %s column\n", path, names[m]);
            fclose(f);
            exit(2);
        }
    }

    int bad = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        char model[64], expl[32], sub[16] = "", lab[16] = "";
        snprintf(model, sizeof model, "%s", field(line, c_model));
        snprintf(expl, sizeof expl, "%s", field(line, c_expl));
        if (!is_control) {
            snprintf(sub, sizeof sub, "%s", field(line, c_sub));
            snprintf(lab, sizeof lab, "%s", field(line, c_lab));
        }
        const long want_n = atol(field(line, c_n));

        Cell *c = NULL;
        for (int i = 0; i < ncells; i++)
            if (strcmp(cells[i].model, model) == 0 && strcmp(cells[i].expl, expl) == 0
                && strcmp(cells[i].sub, sub) == 0 && strcmp(cells[i].lab, lab) == 0)
                c = &cells[i];
        if (!c) {
            printf("  %-24s %-13s %-6s %-8s FAIL no records for this cell\n",
                   model, expl, sub, lab);
            bad++;
            continue;
        }
        if (c->n != want_n) {
            printf("  %-24s %-13s %-6s %-8s FAIL n %ld, published %ld\n",
                   model, expl, sub, lab, c->n, want_n);
            bad++;
            continue;
        }

        int row_bad = 0;
        for (int m = 0; m < nmetric; m++) {
            char got[64];
            if (m == 0) wilson(c->ks, c->n, got, sizeof got);
            else if (m == 1) wilson(c->kl, c->n, got, sizeof got);
            else wilson(c->cv, c->ct, got, sizeof got);
            const char *want = field(line, cols[m]);
            if (strcmp(got, want) != 0) {
                printf("  %-24s %-13s %-6s %-8s FAIL %s: C %s, published %s\n",
                       model, expl, sub, lab, names[m], got, want);
                row_bad++;
            }
            (*checked)++;
        }
        if (!row_bad)
            printf("  %-24s %-13s %-6s %-8s ok  n=%ld\n",
                   model, expl, sub, lab, c->n);
        bad += row_bad;
    }
    fclose(f);
    return bad;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    char path[1024];

    snprintf(path, sizeof path, "%s/reports/counterfactual.json", root);
    const long records = read_records(path);
    printf("read %ld records from counterfactual.json into %d cells\n",
           records, ncells);

    int checked = 0, bad = 0;
    printf("\nreports/counterfactual_summary.csv\n");
    snprintf(path, sizeof path, "%s/reports/counterfactual_summary.csv", root);
    bad += check(path, 0, &checked);

    printf("\nreports/edge_reading_control.csv\n");
    snprintf(path, sizeof path, "%s/reports/edge_reading_control.csv", root);
    bad += check(path, 1, &checked);

    if (bad) {
        printf("\n%d of %d published intervals disagree with C\n", bad, checked);
        return 1;
    }
    printf("\nC reproduces all %d published intervals exactly, "
           "as printed strings\n", checked);
    return 0;
}
