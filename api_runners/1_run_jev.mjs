// Screening with Jev through the TypeSafe API: 10 sequential runs of one protocol.
// Usage: node api_runners/1_run_jev.mjs <protocol> [--dry-run] [--mock]
//   noul-k1                      Jev Noul (newly drafted prompts), one record per request
//   noul-k10 | noul-k25 | noul-k50   Jev Noul with up to 10, 25, or 50 records per request
//   choice-v4                    Jev Choice (prompts reused from the 2024 study)
// Reference labels are not read here; performance is computed afterwards by scripts/recalculate.py.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { ROOT, RESULTS_DIR, DATASETS, RETRY_DELAYS_MS, readJson, sha256, recordText, loadRecords, isTransient, safeError, runPool, sleep } from './common.mjs';

const MODEL = 'jev-1.13.0';
const ENDPOINT = 'https://api.typesafe.ai/v1/systemone';
const RUNS = 10;
const CONCURRENCY = 50;
const SPACING_MS = 55;              // official limit: 1,200 requests per minute
const TOKENS_PER_SEC = 200000;      // official limit: 250,000 input tokens per second
const CHARS_PER_TOKEN = 3.5;        // for pacing only; costs use the token counts returned by the API
const BUNDLE_CHAR_BUDGET = 90000;   // keeps a request within its input budget of about 32,000 tokens
const USD_PER_MILLION_INPUT = 0.042;

const PROTOCOLS = {
  'noul-k1': { kind: 'noul', k: 1, promptFile: 'prompts_noul.json' },
  'noul-k10': { kind: 'noul', k: 10, promptFile: 'prompts_noul.json' },
  'noul-k25': { kind: 'noul', k: 25, promptFile: 'prompts_noul.json' },
  'noul-k50': { kind: 'noul', k: 50, promptFile: 'prompts_noul.json' },
  'choice-v4': { kind: 'choice', k: 1, promptFile: 'prompts_choice.json' },
};

const args = process.argv.slice(2);
const protocol = args.find(a => !a.startsWith('--'));
const flags = args.filter(a => a.startsWith('--'));
if (!PROTOCOLS[protocol] || flags.some(f => !['--dry-run', '--mock'].includes(f))) {
  throw Error(`Usage: node api_runners/1_run_jev.mjs <${Object.keys(PROTOCOLS).join('|')}> [--dry-run] [--mock]`);
}
const { kind, k, promptFile } = PROTOCOLS[protocol];
const dryRun = flags.includes('--dry-run'), mock = flags.includes('--mock');

const records = loadRecords();
const prompts = readJson(path.join(ROOT, 'configuration', promptFile));

// ---- Requests -------------------------------------------------------------

const label = i => 'R' + String(i + 1).padStart(2, '0');

function buildRequest(bundle) {
  const dataset = bundle[0].dataset;
  if (kind === 'noul') {
    const p = prompts[dataset];
    const state = `# Eligibility criteria\n${p.criteria}\n\n# Records\n` + bundle.map((r, i) => `[${label(i)}]\n${recordText(r)}`).join('\n\n');
    const questions = Object.fromEntries(bundle.map((r, i) => [label(i), { type: 'noul', instructions: p.question_template.replace('{label}', label(i)) }]));
    return { state, questions };
  }
  const state = `#Title and Abstract of the study\n${recordText(bundle[0])}`;
  return { state, questions: { screening: { type: 'choice', instructions: prompts[dataset].instructions, criteria: prompts.options } } };
}

// Seeded shuffle: bundle composition and record position vary between runs but remain reproducible.
function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffled(items, seed) {
  const random = seededRandom(seed), a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Requests of one run. A request holds records of one dataset, at most k of them, within the character budget.
function bundlesFor(run) {
  const bundles = [];
  for (const dataset of DATASETS) {
    const own = records.filter(r => r.dataset === dataset);
    const order = k === 1 ? own : shuffled(own, parseInt(sha256(`${protocol}|${dataset}|${run}`).slice(0, 8), 16));
    let current = [], chars = 0;
    for (const record of order) {
      const n = recordText(record).length;
      if (current.length && (current.length >= k || chars + n > BUNDLE_CHAR_BUDGET)) { bundles.push(current); current = []; chars = 0; }
      current.push(record); chars += n;
    }
    if (current.length) bundles.push(current);
  }
  return bundles.map((bundle, index) => ({ index, records: bundle, request: buildRequest(bundle) }));
}

function validate(raw, request) {
  if (raw.model !== MODEL) throw Error('Unexpected model version; stopped to preserve comparability');
  for (const id of Object.keys(request.questions)) {
    const answer = raw.answers?.[id];
    if (kind === 'noul') {
      if (answer?.type !== 'noul' || !Number.isFinite(answer.noul) || answer.noul < 0 || answer.noul > 1) throw Error('Invalid noul answer');
    } else {
      const p = answer?.probabilities;
      if (answer?.type !== 'choice' || !p || !['I', 'E'].every(x => Number.isFinite(p[x]) && p[x] >= 0 && p[x] <= 1)) throw Error('Invalid choice answer');
    }
  }
}

// ---- API call ---------------------------------------------------------------

// Pacing keeps dispatch under both official limits. The wait is not part of latencyMs.
let nextSlot = 0, tokenFree = 0;
async function pace(estimatedTokens) {
  const now = Date.now(), slot = Math.max(now, nextSlot, tokenFree);
  nextSlot = slot + SPACING_MS;
  tokenFree = slot + estimatedTokens / TOKENS_PER_SEC * 1000;
  if (slot > now) await sleep(slot - now);
}

async function mockFetch(_url, init) {
  const body = JSON.parse(init.body);
  await sleep(20);
  const answers = Object.fromEntries(Object.entries(body.questions).map(([id, q]) =>
    [id, q.type === 'noul' ? { type: 'noul', noul: 0.1 } : { type: 'choice', choice: 'E', probabilities: { I: 0.1, E: 0.9 }, confidence: 0.8 }]));
  return { ok: true, status: 200, headers: new Map(), json: async () => ({ model: MODEL, answers, usage: { input_tokens: Math.round(init.body.length / CHARS_PER_TOKEN), output_tokens: 0 } }) };
}

async function call(request, key) {
  await pace(JSON.stringify(request).length / CHARS_PER_TOKEN);
  const began = performance.now(), startedAt = new Date().toISOString();
  const response = await (mock ? mockFetch : fetch)(ENDPOINT, {
    method: 'POST',
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: MODEL, ...request }),
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) {
    const e = new Error(`TypeSafe HTTP ${response.status}`);
    e.statusCode = response.status;
    const v = response.headers.get('retry-after');
    e.retryAfterMs = v ? Math.max(0, Number.isFinite(Number(v)) ? Number(v) * 1000 : Date.parse(v) - Date.now()) : 0;
    throw e;
  }
  const raw = await response.json(), latencyMs = performance.now() - began;
  validate(raw, request);
  return { startedAt, latencyMs, model: raw.model, answers: raw.answers, usage: raw.usage };
}

// ---- Plan ---------------------------------------------------------------------

const run1 = bundlesFor(1);
const plan = {
  protocol, kind, recordsPerRequest: k, runs: RUNS, concurrency: CONCURRENCY, records: records.length, model: MODEL, endpoint: ENDPOINT,
  minimumRequestSpacingMs: SPACING_MS, pacedTokensPerSecond: TOKENS_PER_SEC, bundleCharBudget: BUNDLE_CHAR_BUDGET, inputUsdPerMillion: USD_PER_MILLION_INPUT,
  promptFile, promptSha256: sha256(prompts), recordsSha256: sha256(records), run1RequestSha256: sha256(run1.map(b => b.request)), requestsInRun1: run1.length,
  notes: 'Runs are sequential. latencyMs is measured from request dispatch to parsed response and excludes rate-limit pacing; wallSeconds includes pacing and retries.',
};
if (dryRun) {
  const estimatedInputTokensRun1 = Math.round(run1.reduce((n, b) => n + JSON.stringify(b.request).length, 0) / CHARS_PER_TOKEN);
  console.log(JSON.stringify({ ...plan, apiCalls: 0, estimatedInputTokensRun1, exampleRequest: run1[0].request.state.slice(0, 300) }, null, 2));
  process.exit(0);
}

const key = mock ? 'mock' : process.env.TYPESAFE_API_KEY;
if (!key) throw Error('Set TYPESAFE_API_KEY');

// A finished experiment with the same plan is never repeated; an unfinished one is resumed.
const prefix = (mock ? 'mock-' : '') + `jev-${protocol}-`;
fs.mkdirSync(RESULTS_DIR, { recursive: true });
const prior = fs.readdirSync(RESULTS_DIR).filter(n => n.startsWith(prefix)).sort().at(-1);
const priorDir = prior && path.join(RESULTS_DIR, prior);
const samePlan = prior && ['promptSha256', 'recordsSha256', 'run1RequestSha256', 'model'].every(x => readJson(path.join(priorDir, 'plan.json'))[x] === plan[x]);
const priorComplete = samePlan && fs.existsSync(path.join(priorDir, 'summaries.json')) && readJson(path.join(priorDir, 'summaries.json')).filter(s => s.complete).length === RUNS;
if (priorComplete) { console.log(`${protocol}: already complete (${prior}); skipped.`); process.exit(0); }
const resuming = samePlan && !priorComplete;
const root = resuming ? priorDir : path.join(RESULTS_DIR, prefix + new Date().toISOString().replaceAll(':', '-'));
fs.mkdirSync(root, { recursive: true });
if (resuming) console.log('Resuming ' + root);
else {
  fs.writeFileSync(path.join(root, 'plan.json'), JSON.stringify(plan, null, 2));
  fs.writeFileSync(path.join(root, 'prompts-frozen.json'), JSON.stringify(prompts, null, 2));
  fs.copyFileSync(fileURLToPath(import.meta.url), path.join(root, 'runner-snapshot.mjs'));
}

// ---- Runs ---------------------------------------------------------------------

const summaries = [];
for (let run = 1; run <= RUNS; run++) {
  const folder = path.join(root, `run-${String(run).padStart(2, '0')}`);
  fs.mkdirSync(folder, { recursive: true });
  const summaryFile = path.join(folder, 'summary.json'), responseFile = path.join(folder, 'responses.jsonl');
  const old = fs.existsSync(summaryFile) ? readJson(summaryFile) : null;
  if (old?.complete) { summaries.push(old); continue; }

  const bundles = bundlesFor(run);
  const rows = fs.existsSync(responseFile) ? fs.readFileSync(responseFile, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [];
  const done = new Set(rows.map(r => r.bundle));
  if (done.size !== rows.length) throw Error('Duplicate saved bundle');
  let attempts = old?.attempts ?? rows.length, permanentFailure = false;
  const start = performance.now();
  console.log(`${protocol} run ${run}/${RUNS}: ${bundles.length} requests`);

  await runPool(bundles.filter(b => !done.has(b.index)), CONCURRENCY, async b => {
    for (let attempt = 1; attempt <= 5; attempt++) {
      attempts++;
      try {
        const result = await call(b.request, key);
        const row = { bundle: b.index, dataset: b.records[0].dataset, ids: b.records.map(r => r.id), attempt, ...result };
        fs.appendFileSync(responseFile, JSON.stringify(row) + '\n');
        rows.push(row);
        if (rows.length % 500 === 0) console.log(`  ${rows.length}/${bundles.length}`);
        return true;
      } catch (e) {
        const retryable = isTransient(e);
        fs.appendFileSync(path.join(folder, 'errors.jsonl'), JSON.stringify({ bundle: b.index, attempt, at: new Date().toISOString(), details: safeError(e, key) }) + '\n');
        if (!retryable || attempt === 5) {
          if (!retryable) permanentFailure = true;
          console.error(`Stopped at bundle ${b.index}; see errors.jsonl`);
          return false;
        }
        await sleep(Math.max(RETRY_DELAYS_MS[attempt - 1], e.retryAfterMs || 0));
      }
    }
  });

  const inputTokens = rows.reduce((n, r) => n + (r.usage?.input_tokens ?? 0), 0);
  const summary = {
    rep: run, complete: rows.length === bundles.length, requests: rows.length, records: rows.reduce((n, r) => n + r.ids.length, 0), attempts,
    wallSeconds: (old?.wallSeconds ?? 0) + (performance.now() - start) / 1000, resumed: done.size > 0, inputTokens, estimatedCostUsd: inputTokens / 1e6 * USD_PER_MILLION_INPUT,
  };
  summaries.push(summary);
  fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 2));
  fs.writeFileSync(path.join(root, 'summaries.json'), JSON.stringify(summaries, null, 2));
  console.log(`  done: ${summary.records} records in ${summary.wallSeconds.toFixed(1)} s`);
  if (!summary.complete) { process.exitCode = permanentFailure ? 2 : 75; break; } // 75: temporary failure; launch again to resume
}
console.log('Saved: ' + root);
