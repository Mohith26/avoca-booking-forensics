// Run: node tests/core.test.js   (no dependencies)
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const F = require('../core.js');

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); passed++; console.log('  ok  ' + name); }
  catch (e) { failed++; console.log('  FAIL ' + name + '\n       ' + e.message); }
}
const root = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(root, 'data/calls.json')));
const cf = JSON.parse(fs.readFileSync(path.join(root, 'data/counterfactuals.json')));
const calls = data.calls;

test('tagger: agent acts', () => {
  assert.deepStrictEqual(F.tagTurn('agent', "I'm not able to quote pricing over the phone, but the technician will go over pricing with you on site."), ['PRICE_DEFLECT']);
  assert.deepStrictEqual(F.tagTurn('agent', "I'm sorry, we don't have any openings Friday 11 to 2."), ['NO_SLOT']);
  assert.ok(F.tagTurn('agent', 'Our diagnostic visit is $89, and that is credited toward the repair.').includes('PRICE_QUOTE'));
  assert.ok(F.tagTurn('agent', 'Perfect, thanks. Can I get the service address?').includes('INFO_CAPTURED'));
});
test('tagger: caller cues and reactions', () => {
  assert.ok(F.tagTurn('caller', "Hi, there's no heat and I have a newborn at home.").includes('URGENT'));
  assert.ok(F.tagTurn('caller', 'Hmm. Okay... You know, let me think about it and I\'ll call back.').includes('HESITANT'));
  assert.ok(F.tagTurn('caller', "I'd really like a number before someone comes out. I'll call around.").includes('EXIT'));
  assert.deepStrictEqual(F.tagTurn('caller', 'Okay, that is fair.'), []);
});
test('normInv matches known quantiles', () => {
  assert.ok(Math.abs(F.normInv(0.975) - 1.959964) < 1e-4);
  assert.ok(Math.abs(F.normInv(0.8) - 0.841621) < 1e-4);
  assert.ok(Math.abs(F.normInv(0.5)) < 1e-9);
});
test('IRLS logistic regression recovers planted coefficients', () => {
  const rnd = F.mulberry32(7), wTrue = [-0.5, 1.2, -2.0];
  const X = [], y = [];
  for (let i = 0; i < 40000; i++) {
    const x = [1, rnd() < 0.5 ? 1 : 0, rnd() < 0.3 ? 1 : 0];
    X.push(x); y.push(rnd() < F.predict(wTrue, x) ? 1 : 0);
  }
  const w = F.fitLogistic(X, y, 0.01);
  w.forEach((v, j) => assert.ok(Math.abs(v - wTrue[j]) < 0.12, `w[${j}]=${v.toFixed(3)} vs ${wTrue[j]}`));
});
test('two-proportion A/B sample size matches the textbook formula', () => {
  const n = F.abSampleSize(0.70, 0.75);
  assert.ok(n >= 1240 && n <= 1260, 'n=' + n);
  assert.ok(F.abSampleSize(0.70, 0.80) < n);
});
test('PPI++ is unbiased and covers under a biased judge', () => {
  const rnd = F.mulberry32(11);
  let sum = 0, hit = 0, wP = 0, wH = 0;
  const reps = 400, truth = 0.62;
  for (let r = 0; r < reps; r++) {
    const make = (n) => Array.from({ length: n }, () => {
      const y = rnd() < truth ? 1 : 0;
      const f = Math.min(1, Math.max(0, 0.25 + 0.5 * y + 0.15 + (rnd() - 0.5) * 0.4)); // biased upward
      return { y, f };
    });
    const L = make(40), U = make(400);
    const e = F.ppiMean(L.map((d) => d.f), L.map((d) => d.y), U.map((d) => d.f));
    sum += e.est;
    const se = Math.sqrt(e.v);
    if (Math.abs(e.est - truth) <= 1.96 * se) hit++;
    wP += 2 * 1.96 * se; wH += 2 * 1.96 * Math.sqrt(truth * (1 - truth) / 40);
  }
  assert.ok(Math.abs(sum / reps - truth) < 0.01, 'mean est ' + (sum / reps));
  assert.ok(hit / reps >= 0.92, 'coverage ' + hit / reps);
  assert.ok(wP < wH, 'PPI interval should be narrower than labels-only');
});

const model = F.trainValueModel(calls);
const lost = calls.filter((c) => !c.booked);
const results = lost.map((c) => ({ call: c, ...F.localize(c, model.w) }));

test('value model: a full transcript of a booked call ends with high V', () => {
  const b = calls.find((c) => c.booked);
  const r = F.localize(b, model.w);
  assert.ok(r.V[r.V.length - 1] > 0.85, 'V_end=' + r.V[r.V.length - 1]);
});
test('blame localization recovers the planted failure mode (>= 90%)', () => {
  const acc = results.filter((r) => r.mode === r.call.truth.mode).length / results.length;
  assert.ok(acc >= 0.9, 'mode accuracy ' + acc.toFixed(3));
});
test('blame localization finds the planted turn and beats "last agent turn"', () => {
  const agentCaused = results.filter((r) => r.call.truth.mode !== 'CALLER_SHOPPING');
  const acc = agentCaused.filter((r) => r.blameTurn === r.call.truth.turn).length / agentCaused.length;
  const base = agentCaused.filter((r) => F.lastAgentTurnBefore(r.call) === r.call.truth.turn).length / agentCaused.length;
  assert.ok(acc >= 0.9, 'turn accuracy ' + acc.toFixed(3));
  assert.ok(acc > base + 0.15, `model ${acc.toFixed(3)} vs baseline ${base.toFixed(3)}`);
});
test('failure modes: counts and job value add up', () => {
  const modes = F.failureModes(results);
  assert.strictEqual(modes.reduce((s, m) => s + m.count, 0), lost.length);
  assert.strictEqual(modes.reduce((s, m) => s + m.value, 0), lost.reduce((s, c) => s + c.value, 0));
});

function strataFor(idx) {
  const st = {};
  idx.forEach((i) => {
    const c = calls[i];
    const k = c.booked ? 'booked' : 'lost:' + F.localize(c, model.w).mode;
    (st[k] = st[k] || []).push(i);
  });
  return st;
}
const allIdx = calls.map((_, i) => i);
const strata = strataFor(allIdx);
test('design is deterministic for a seed and respects the budget', () => {
  const O = cf.outcomes.quote_fee;
  const ops = { judge: (i) => O[i][1], human: (i) => O[i][0], diverged: (i) => O[i][2] === 1, observed: (i) => (calls[i].booked ? 1 : 0) };
  const a = F.designSample(strata, 400, 60, 5, ops), b = F.designSample(strata, 400, 60, 5, ops);
  assert.deepStrictEqual(a, b);
  const n = Object.values(a).reduce((s, x) => s + x.same.length + x.labeled.length + x.unlabeled.length, 0);
  assert.ok(n <= 400, 'replays ' + n);
});
for (const cand of cf.candidates) {
  test(`replay estimate for "${cand.short}": PPI++ 95% CI covers the simulated truth (>= 90% of 200 resamples)`, () => {
    const O = cf.outcomes[cand.id];
    const ops = { judge: (i) => O[i][1], human: (i) => O[i][0], diverged: (i) => O[i][2] === 1, observed: (i) => (calls[i].booked ? 1 : 0) };
    const truth = O.reduce((s, r) => s + r[0], 0) / O.length;
    const cov = F.coverage(strata, 400, 60, ops, truth, 200, 3);
    assert.ok(cov.ppi.coverage >= 0.9, 'coverage ' + cov.ppi.coverage);
    assert.ok(cov.ppi.width < cov.human.width, 'PPI should be narrower than labels-only');
  });
}
test('judge-only intervals are overconfident for the price change (documents judge bias)', () => {
  const O = cf.outcomes.quote_fee;
  const ops = { judge: (i) => O[i][1], human: (i) => O[i][0], diverged: (i) => O[i][2] === 1, observed: (i) => (calls[i].booked ? 1 : 0) };
  const truth = O.reduce((s, r) => s + r[0], 0) / O.length;
  const cov = F.coverage(strata, 400, 60, ops, truth, 200, 3);
  assert.ok(cov.judge.coverage < 0.5, 'judge coverage ' + cov.judge.coverage);
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
