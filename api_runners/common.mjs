// Helpers shared by the Jev runner and the LLM runner.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const RESULTS_DIR = path.join(ROOT, 'results', 'api_runs');
export const DATASETS = ['takeshima', 'maruki']; // light therapy; adjunctive pharmacotherapy (first authors of the source reviews)

export const readJson = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^﻿/, ''));
export const sha256 = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');

// Only the title and abstract of a record are ever sent to a model.
export const recordText = record => `Title: ${record.title}\nAbstract: ${record.abstract}`;

export function loadRecords() {
  const file = path.join(ROOT, '1_data', 'records.json');
  if (!fs.existsSync(file)) throw Error('Original input titles and abstracts are not distributed. API inference requires separately obtained 1_data/records.json and appropriate access rights; see 1_data/README.md. Saved-output reanalysis uses scripts/recalculate.py and does not require this file.');
  const records = readJson(file);
  if (new Set(records.map(r => r.id)).size !== records.length) throw Error('Record ids are not unique');
  if (records.some(r => !DATASETS.includes(r.dataset))) throw Error('Unknown dataset name');
  return records;
}

export function isTransient(error) {
  for (let e = error, depth = 0; e && depth < 5; e = e.cause, depth++) {
    if ([408, 429, 500, 502, 503, 504, 529].includes(e.statusCode)) return true;
    if (['TimeoutError', 'AbortError'].includes(e.name)) return true;
    if (['ECONNRESET', 'ETIMEDOUT', 'EAI_AGAIN', 'UND_ERR_CONNECT_TIMEOUT', 'UND_ERR_SOCKET'].includes(e.code)) return true;
  }
  return false;
}
export const RETRY_DELAYS_MS = [5000, 10000, 20000, 40000]; // up to five attempts per request

// Error details for errors.jsonl, with the API key removed.
export function safeError(error, secret = '') {
  const clean = value => {
    let text = String(value ?? '');
    if (secret) text = text.replaceAll(secret, '[REDACTED]');
    return text.replace(/Bearer\s+[^\s"'<>]+/gi, 'Bearer [REDACTED]').slice(0, 2000);
  };
  const details = [];
  for (let e = error, depth = 0; e && depth < 4; depth++, e = e.cause) {
    details.push({ name: clean(e.name), message: clean(e.message), statusCode: Number.isInteger(e.statusCode) ? e.statusCode : null, body: clean(e.body || '') });
  }
  return details;
}

// Runs `work` over `items` with at most `concurrency` in flight; stops dispatching once `work` returns false.
export async function runPool(items, concurrency, work) {
  let next = 0, stopped = false;
  const worker = async () => {
    while (!stopped && next < items.length) {
      if (await work(items[next++]) === false) stopped = true;
    }
  };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
}

export const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
