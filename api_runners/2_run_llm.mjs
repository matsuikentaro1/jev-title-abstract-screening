// Generative LLM comparator through the OpenAI API: one include/exclude decision per record, one run per launch.
// The message holds the record and the criteria, query, and response options used with Jev Choice.
// Usage: node api_runners/2_run_llm.mjs <model id> [--effort=default] [--rep=N] [--dry-run] [--mock]
//   --effort   omitted: the lowest reasoning effort the API accepts; default: the model's default reasoning
//   --rep      repetition number; repetitions 2 and above are saved in their own result folder
// Conditions reported: gpt-5-mini; gpt-5-mini --effort=default; gpt-6-astra --effort=default (three repetitions each).
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { ROOT, RESULTS_DIR, RETRY_DELAYS_MS, readJson, sha256, recordText, loadRecords, isTransient, safeError, runPool, sleep } from './common.mjs';

const BASE = 'https://api.openai.com/v1';
const ENDPOINT = BASE + '/chat/completions';
const CONCURRENCY = 50;

const args = process.argv.slice(2);
const option = name => (args.find(a => a.startsWith(`--${name}=`)) || '').split('=')[1] || '';
const modelId = args.find(a => !a.startsWith('--'));
const effort = option('effort'), rep = Number(option('rep')) || 1;
const dryRun = args.includes('--dry-run'), mock = args.includes('--mock');
if (!modelId) throw Error('Usage: node api_runners/2_run_llm.mjs <model id> [--effort=default] [--rep=N] [--dry-run] [--mock]');

const prompts = readJson(path.join(ROOT, 'configuration', 'prompts_choice.json'));
const records = loadRecords();

// ---- Requests -------------------------------------------------------------

const OPTIONS = `Response options\nI: ${prompts.options.I}\nE: ${prompts.options.E}\n\nAnswer with the single letter I or E only.`;
const message = record => `#Title and Abstract of the study\n${recordText(record)}\n\n${prompts[record.dataset].instructions}\n\n${OPTIONS}`;

function buildBody(record, p) {
  const body = { model: modelId, messages: [{ role: 'user', content: message(record) }] };
  if (p.temperature !== undefined) body.temperature = p.temperature;
  if (p.reasoning_effort !== undefined) body.reasoning_effort = p.reasoning_effort;
  if (p.logprobs) { body.logprobs = true; body.top_logprobs = 5; }
  return body;
}

// Decision: first letter of the reply. Any other reply is saved with decision null (invalid).
function parseDecision(text) {
  const m = String(text ?? '').trim().match(/^[\s"'*(\[]*([IE])\b/i);
  return m ? m[1].toUpperCase() : null;
}

// Probability of "I" from the token log-probabilities of the first answer token, when the API returns them.
function probabilityOfInclude(choice) {
  const top = choice?.logprobs?.content?.find(t => /^[\s"']*[IE]/i.test(t.token))?.top_logprobs;
  if (!top) return null;
  let pI = 0, pE = 0;
  for (const t of top) {
    const letter = t.token.trim().replace(/^["']/, '').toUpperCase();
    if (letter === 'I') pI += Math.exp(t.logprob);
    else if (letter === 'E') pE += Math.exp(t.logprob);
  }
  return pI + pE > 0 ? pI / (pI + pE) : null;
}

// ---- API call ---------------------------------------------------------------

async function mockFetch() {
  await sleep(15);
  return { ok: true, status: 200, headers: new Map(), json: async () => ({ model: modelId, choices: [{ message: { content: 'E' }, finish_reason: 'stop' }], usage: { prompt_tokens: 1000, completion_tokens: 1 } }) };
}

async function post(body, key) {
  const began = performance.now(), startedAt = new Date().toISOString();
  const res = await (mock ? mockFetch : fetch)(ENDPOINT, {
    method: 'POST',
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(180000),
  });
  if (!res.ok) {
    const e = new Error(`HTTP ${res.status}`);
    e.statusCode = res.status;
    try { e.body = await res.text(); } catch {}
    const ms = res.headers.get('retry-after-ms'), v = res.headers.get('retry-after');
    e.retryAfterMs = ms ? Math.max(0, Number(ms) || 0) : (v ? Math.max(0, Number(v) * 1000 || 0) : 0);
    throw e;
  }
  return { raw: await res.json(), latencyMs: performance.now() - began, startedAt };
}

// Parameter negotiation on the first record, in a fixed order of preference: temperature 0, the requested reasoning
// effort (or the lowest accepted), and token log-probabilities. What the API rejects is dropped and recorded.
async function negotiate(key) {
  const efforts = effort ? [effort === 'default' ? undefined : effort] : ['none', 'minimal', 'low', undefined];
  const candidates = [];
  for (const temperature of [0, undefined]) for (const reasoning_effort of efforts) for (const logprobs of [true, false]) candidates.push({ temperature, reasoning_effort, logprobs });
  const rejected = [];
  for (const candidate of candidates) {
    try {
      await post(buildBody(records[0], candidate), key);
      return { accepted: candidate, rejected };
    } catch (e) {
      if (![400, 403, 422].includes(e.statusCode)) throw e;
      rejected.push({ ...candidate, message: String(e.body || e.message).slice(0, 300) });
    }
  }
  throw Error('No parameter combination was accepted. Last message: ' + (rejected.at(-1)?.message || ''));
}

// ---- Plan ---------------------------------------------------------------------

const plan = {
  model: modelId, requestedReasoningEffort: effort || 'lowest accepted', repetition: rep, endpoint: ENDPOINT, records: records.length, concurrency: CONCURRENCY,
  promptFile: 'prompts_choice.json', promptSha256: sha256(prompts), recordsSha256: sha256(records), requestSha256: sha256(records.map(message)),
  notes: 'One request per record. latencyMs excludes retry waits; wallSeconds includes them.',
};
if (dryRun) {
  const estimatedInputTokens = Math.round(records.reduce((n, r) => n + message(r).length, 0) / 3.8);
  console.log(JSON.stringify({ ...plan, apiCalls: 0, estimatedInputTokens, exampleMessage: message(records[0]).slice(0, 1200) }, null, 2));
  process.exit(0);
}

const key = mock ? 'mock' : process.env.OPENAI_API_KEY;
if (!key) throw Error('Set OPENAI_API_KEY');

// A finished run with the same requests is never repeated; an unfinished one is resumed.
const slug = modelId.replace(/[^a-z0-9.-]+/gi, '_');
const prefix = (mock ? 'mock-' : '') + `llm-${slug}-` + (effort ? `effort_${effort}-` : '') + (rep > 1 ? `rep${rep}-` : '');
fs.mkdirSync(RESULTS_DIR, { recursive: true });
const prior = fs.readdirSync(RESULTS_DIR).filter(n => n.startsWith(prefix) && /^\d{4}-/.test(n.slice(prefix.length)) && fs.existsSync(path.join(RESULTS_DIR, n, 'plan.json'))).sort().at(-1);
const priorDir = prior && path.join(RESULTS_DIR, prior);
const sameRequests = prior && readJson(path.join(priorDir, 'plan.json')).requestSha256 === plan.requestSha256;
const priorComplete = prior && fs.existsSync(path.join(priorDir, 'summary.json'));
if (sameRequests && priorComplete) { console.log(`${modelId}: already complete (${prior}); skipped.`); process.exit(0); }
const resuming = sameRequests && !priorComplete;

if (!mock) { // the model id is confirmed against the API's own list
  const res = await fetch(BASE + '/models', { headers: { Authorization: `Bearer ${key}` } });
  if (!res.ok) throw Error('Could not list models: HTTP ' + res.status);
  if (!(await res.json()).data.some(m => m.id === modelId)) throw Error('Model id not found: ' + modelId);
}

const root = resuming ? priorDir : path.join(RESULTS_DIR, prefix + new Date().toISOString().replaceAll(':', '-'));
fs.mkdirSync(root, { recursive: true });
let params;
if (resuming) {
  params = readJson(path.join(root, 'plan.json')).parameters;
  console.log('Resuming ' + root);
} else {
  const negotiated = await negotiate(key);
  params = negotiated.accepted;
  fs.writeFileSync(path.join(root, 'plan.json'), JSON.stringify({ ...plan, parameters: params, rejectedParameters: negotiated.rejected }, null, 2));
  fs.copyFileSync(fileURLToPath(import.meta.url), path.join(root, 'runner-snapshot.mjs'));
  console.log('Parameters accepted by the API:', JSON.stringify(params));
}

// ---- Run ------------------------------------------------------------------------

const responseFile = path.join(root, 'responses.jsonl'), timingFile = path.join(root, 'timing.json');
const rows = fs.existsSync(responseFile) ? fs.readFileSync(responseFile, 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse) : [];
const done = new Set(rows.map(r => r.id));
const previousSeconds = fs.existsSync(timingFile) ? readJson(timingFile).wallSeconds : 0;
let attempts = rows.length, permanentFailure = false, rateLimited = 0;
const start = performance.now();
const elapsed = () => previousSeconds + (performance.now() - start) / 1000;

await runPool(records.filter(r => !done.has(r.id)), CONCURRENCY, async record => {
  for (let attempt = 1, limited = 0; attempt <= 5; attempt++) {
    attempts++;
    try {
      const { raw, latencyMs, startedAt } = await post(buildBody(record, params), key);
      const choice = raw.choices?.[0], text = choice?.message?.content ?? '';
      const row = { id: record.id, dataset: record.dataset, attempt, startedAt, latencyMs, model: raw.model, text: String(text).slice(0, 200), decision: parseDecision(text), pInclude: probabilityOfInclude(choice), finish: choice?.finish_reason ?? null, usage: raw.usage ?? null };
      fs.appendFileSync(responseFile, JSON.stringify(row) + '\n');
      rows.push(row);
      if (rows.length % 250 === 0) { console.log(`  ${rows.length}/${records.length}`); fs.writeFileSync(timingFile, JSON.stringify({ wallSeconds: elapsed(), attempts })); }
      return true;
    } catch (e) {
      // Rate-limit responses are waited out (the wait counts towards wallSeconds) and do not use up an attempt.
      if (e.statusCode === 429 && limited < 200 && !/insufficient_quota|billing/i.test(e.body || '')) {
        limited++; attempt--; rateLimited++;
        await sleep(Math.max(e.retryAfterMs || 0, 1000) + Math.random() * 1000);
        continue;
      }
      const retryable = isTransient(e) && e.statusCode !== 429;
      fs.appendFileSync(path.join(root, 'errors.jsonl'), JSON.stringify({ id: record.id, attempt, at: new Date().toISOString(), details: safeError(e, key) }) + '\n');
      if (!retryable || attempt === 5) {
        if (!retryable) permanentFailure = true;
        console.error(`Stopped at ${record.id}; see errors.jsonl`);
        return false;
      }
      await sleep(Math.max(RETRY_DELAYS_MS[attempt - 1], e.retryAfterMs || 0));
    }
  }
});

const wallSeconds = elapsed();
fs.writeFileSync(timingFile, JSON.stringify({ wallSeconds, attempts }));
if (rows.length !== records.length) {
  process.exitCode = permanentFailure ? 2 : 75; // 75: temporary failure; launch again to resume
  console.log(`Incomplete: ${rows.length}/${records.length}. Launch again to resume.`);
} else {
  const sum = field => rows.reduce((n, r) => n + (r.usage?.[field] ?? 0), 0);
  fs.writeFileSync(path.join(root, 'summary.json'), JSON.stringify({
    model: modelId, records: rows.length, wallSeconds, attempts, rateLimitedResponses: rateLimited, resumed: done.size > 0,
    invalidReplies: rows.filter(r => !r.decision).length, withTokenProbabilities: rows.filter(r => r.pInclude != null).length,
    promptTokens: sum('prompt_tokens'), completionTokens: sum('completion_tokens'),
  }, null, 2));
  console.log(`Done: ${rows.length} records in ${wallSeconds.toFixed(1)} s. Saved: ${root}`);
}
