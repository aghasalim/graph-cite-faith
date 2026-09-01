# The numbers the README states in prose rather than in a table.
#
# verify/readme_tables.mjs covers the three markdown tables. The sentences
# around them carry a dozen more figures that appear in no table and in no CSV:
# the citation totals, the parse-failure rate, how many narrations the run is
# built on, how far the qwen fabrication goes, the motif recovery of the two
# explainers. Those are the numbers in the abstract and the headline, so they
# are the ones a reader takes away, and they are the easiest to leave stale
# after a rerun.
#
# Each check recomputes a figure from reports/counterfactual.json, renders the
# sentence fragment the README ought to contain, and requires that exact string
# to be present. The fragment is built from the data rather than compared to a
# constant, so if the data moves the expected string moves with it and the grep
# fails. Ruby because this is text handling, and the standard library has the
# JSON reader and the regexes already.
#
# Run: ruby verify/prose.rb <root>

require 'json'

root = ARGV[0] || '.'
records = JSON.parse(File.read(File.join(root, 'reports', 'counterfactual.json'),
                               encoding: 'UTF-8'))
# Line wrapping in the README is not a difference in what it says, so the
# whole document is collapsed to single spaces before anything is looked for
# in it. A sentence that runs over three lines still has to be present.
# Read as UTF-8 explicitly rather than in whatever the locale happens to be:
# the README has a multiplication sign in it and CI does not set LANG.
readme = File.read(File.join(root, 'README.md'), encoding: 'UTF-8').gsub(/\s+/, ' ')

failures = 0
checked = 0

# The README writes thousands with a comma.
def grouped(n)
  n.to_s.reverse.scan(/\d{1,3}/).join(',').reverse
end

def wilson(k, n, z = 1.96)
  p = k.to_f / n
  den = 1 + z * z / n.to_f
  centre = (p + z * z / (2.0 * n)) / den
  half = z * Math.sqrt(p * (1 - p) / n + z * z / (4.0 * n * n)) / den
  format('%.3f [%.3f,%.3f]', p, [0.0, centre - half].max, [1.0, centre + half].min)
end

# want: the sentence fragment the recomputed numbers imply. The README has to
# contain it verbatim.
def check(readme, what, want)
  if readme.include?(want)
    puts format('  %-38s %s', what, want)
    true
  else
    puts format('  FAIL %-33s the records give "%s", which the README does not say', what, want)
    false
  end
end

results = []
def add(results, what, want)
  results << [what, want]
end

narrations = records.select { |r| r['kind'] == 'narrate' }
controls = records.select { |r| r['kind'] == 'control' }

# How much evidence the run rests on, and how much of it the parser lost.
add(results, 'run size',
    "#{grouped(narrations.size)} narrations plus #{grouped(controls.size)} control probes")
unparsed = records.count { |r| !r['parsed'] }
add(results, 'parse failure rate',
    format('Parse failures are now %.1f%%.', 100.0 * unparsed / records.size))

# Citation validity. The headline claim is that four of the five models cited
# nothing that was not in front of them; which four is derived, not assumed.
by_model = narrations.group_by { |r| r['model'] }
clean = by_model.select { |_, rs| rs.sum { |r| r['n_cited'] - r['cited_valid'] } == 0 }
dirty = by_model.keys - clean.keys
if clean.size != 4 || dirty.size != 1
  puts "  FAIL #{clean.size} models fabricated nothing and #{dirty.size} did; the README's " \
       'headline is about four and one'
  failures += 1
end
cited = clean.values.flatten.sum { |r| r['n_cited'] }
valid = clean.values.flatten.sum { |r| r['cited_valid'] }
add(results, 'clean models, cited ids',
    "#{grouped(valid)} of #{grouped(cited)} cited node ids were real across four of five models")
add(results, 'clean models, restated', "Across #{grouped(cited)} cited node ids")

# The one model that did fabricate, and how far it goes.
unless dirty.empty?
  rs = by_model[dirty.first]
  n_cited = rs.sum { |r| r['n_cited'] }
  n_valid = rs.sum { |r| r['cited_valid'] }
  bad_narrations = rs.count { |r| r['n_cited'] > r['cited_valid'] }
  short = dirty.first.split('/').last
  add(results, 'the fabricating model',
      "#{short} fabricated #{n_cited - n_valid} node ids out of #{n_cited} " \
      "(#{wilson(n_valid, n_cited)}), across #{bad_narrations} of its #{rs.size} narrations")
end

# The contrast the paper is built on: citation validity pinned at its ceiling
# while structure agreement swings across half its range.
cells = narrations.group_by { |r| [r['model'], r['explainer'], r['subgraph'], r['label']] }
validity = cells.values.map { |rs| wilson(rs.sum { |r| r['cited_valid'] }, rs.sum { |r| r['n_cited'] }) }
structure = cells.values.map { |rs| rs.count { |r| r['agrees_with_structure'] }.to_f / rs.size }
add(results, 'citation validity floor and ceiling',
    format('It never drops below %.3f and sits at exactly 1.000 in %d of %d cells',
           validity.map { |s| s[0, 5].to_f }.min,
           validity.count { |s| s.start_with?('1.000') }, validity.size))
add(results, 'structure agreement range',
    format('structure agreement over the same narrations spans %.3f to %.3f',
           structure.min, structure.max))

# The capability spread the abstract opens with, and which model sits at each
# end of it.
reading = controls.select { |r| r['explainer'] == 'gnnexplainer' }
               .group_by { |r| r['model'] }
               .map { |m, rs| [m, rs.count { |r| r['agrees_with_structure'] }.to_f / rs.size] }
worst = reading.min_by { |_, a| a }
best = reading.max_by { |_, a| a }
add(results, 'edge-reading spread',
    format('edge-reading accuracy ranges from %.2f, chance, for %s to %.2f for %s',
           worst[1], 'Llama-3.3-70B', best[1], 'GPT-OSS-20B'))
if worst[0] != 'llama-3.3-70b-versatile' || best[0] != 'openai/gpt-oss-20b'
  puts "  FAIL the extremes are now #{worst[0]} and #{best[0]}, not llama-3.3-70b and gpt-oss-20b"
  failures += 1
end

# The explainer arm. Recovery is a property of the subgraph, so it is read once
# per node rather than once per narration, and only over the 50 nodes the
# saliency arm actually covers.
recovery = {}
records.each do |r|
  next unless r['subgraph'] == 'true'

  (recovery[r['explainer']] ||= {})[r['node']] = r['motif_edges_recovered']
end
if recovery.key?('saliency') && recovery.key?('gnnexplainer')
  nodes = recovery['saliency'].keys
  mean = lambda { |h| h.values_at(*nodes).sum / nodes.size.to_f }
  add(results, 'motif recovery, both explainers',
      format('%.3f of its edges for GNNExplainer and %.3f for saliency',
             mean.call(recovery['gnnexplainer']), mean.call(recovery['saliency'])))
end

# The sharpest single number in the repository: one model never moved.
flat = narrations.select { |r| r['model'] == 'llama-3.1-8b-instant' && r['explainer'] == 'gnnexplainer' }
pairs = flat.group_by { |r| [r['node'], r['subgraph']] }
            .values.select { |g| g.size == 2 }
moved = pairs.count { |g| g[0]['motif_claimed'] != g[1]['motif_claimed'] }
add(results, 'the model that never moved', "over #{pairs.size} flipped pairs")
if moved != 0
  puts "  FAIL llama-3.1-8b moved on #{moved} pairs; the README says it never did"
  failures += 1
end

puts 'README prose against the records'
results.each do |what, want|
  checked += 1
  failures += 1 unless check(readme, what, want)
end

if failures > 0
  puts "\n#{failures} prose checks failed"
  exit 1
end
puts "\nRuby recomputes #{checked} figures the README states in prose and finds " \
     "each one\nstill written there"
