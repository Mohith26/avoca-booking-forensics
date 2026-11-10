/*
 * Booking Forensics core: turn-level blame for lost bookings + a
 * judge-calibrated (prediction-powered) estimate of a proposed change's lift.
 *
 * Pure functions, no dependencies. Works in the browser (window.Forensics)
 * and in Node (module.exports) so the same code is unit-tested and rendered.
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------ utils
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
  function variance(a) { // unbiased sample variance
    if (a.length < 2) return 0;
    const m = mean(a);
    return a.reduce((s, v) => s + (v - m) * (v - m), 0) / (a.length - 1);
  }
  function covariance(a, b) {
    if (a.length < 2) return 0;
    const ma = mean(a), mb = mean(b);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb);
    return s / (a.length - 1);
  }
  function shuffle(arr, rnd) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }
  // inverse standard normal CDF (Acklam), for z-quantiles
  function normInv(p) {
    const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
    const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
    const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
    const pl = 0.02425;
    let q, r;
    if (p < pl) { q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - pl) { q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    q = p - 0.5; r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
  }

  // ------------------------------------------------------------------ 1. turn tagger
  // Rule-based dialogue-act tagger over transcript text. In production this
  // would be an Oversight annotation / small classifier; rules keep the demo
  // dependency-free and auditable.
  const AGENT_RULES = [
    ['EMERGENCY_ROUTE', /emergency (visit|dispatch)|treat that as an emergency|out to you today/i],
    ['STANDARD_VISIT', /standard repair visit|regular service appointment/i],
    ['PRICE_QUOTE', /\$\d+ (service fee|emergency)|diagnostic visit is \$\d+|\$\d+ service fee/i],
    ['PRICE_DEFLECT', /not able to quote pricing|provide a quote when they arrive|go over pricing with you on site/i],
    ['ADDR_REPEAT', /(repeat|spell) (the street|that address|the address|the street name)|address one more time/i],
    ['INFO_CAPTURED', /^(thank you\. i have that|perfect, thanks)/i],
    ['ALT_OFFERED', /closest openings|cancellation waitlist/i],
    ['NO_SLOT', /don't have any openings|don't have that window/i],
    ['SLOT_OK', /^yes, i have .* available/i],
    ['BOOK_ERROR', /trouble (accessing|with) our schedul/i],
  ];
  const CALLER_RULES = [
    ['URGENT', /no heat|newborn|freezing|through the ceiling|burst|water all over|sewage|leaking all over|no hot water|sparked|burning smell|smoking|buzzing and warm/i],
    ['MILD_CUE', /getting pretty cold|little water on the floor|outlets felt warm/i],
    ['PRICE_Q', /how much|what do you charge|what the price is/i],
    ['SHOPPING', /calling around to compare|couple of quotes/i],
    // reactions: evidence of how the previous agent turn landed
    ['HESITANT', /^(hmm\. okay|i guess\. okay|uh, okay|hm, alright)/i],
    ['EXIT', /call around|never mind|i'll pass|call (you )?back|another company|call someone|forget it|wait it out|think about it|look elsewhere|hold off|whatever\. bye|not booking|comparing quotes|other companies|neither of those works|something came up|figure something else/i],
  ];
  function tagTurn(speaker, text) {
    const rules = speaker === 'agent' ? AGENT_RULES : CALLER_RULES;
    const tags = [];
    // agent turns may hold several sentences; tag each sentence-level clause
    for (const [tag, re] of rules) {
      if (tag === 'INFO_CAPTURED' || tag === 'SLOT_OK') {
        // anchored rules: test each sentence start
        const sents = text.split(/(?<=[.?!])\s+/);
        if (sents.some((s) => re.test(s))) tags.push(tag);
      } else if (re.test(text)) tags.push(tag);
    }
    return tags;
  }

  // ------------------------------------------------------------------ 2. prefix value model
  // V(prefix) = P(booked | everything said so far). Caller reactions
  // (hesitation, objections, exits) are the evidence of where a call turned;
  // the localizer attributes each change in V to the agent turn that the
  // caller was reacting to, so a late generic exit after an earlier hesitant
  // reaction is still blamed on the earlier turn.
  const FEATURES = [
    'bias', 'cust:ridgeline', 'cust:copperline',
    'URGENT', 'MILD_CUE', 'PRICE_Q', 'SHOPPING', 'HESITANT', 'EXIT',
    'EMERGENCY_ROUTE', 'STANDARD_VISIT', 'PRICE_QUOTE', 'PRICE_DEFLECT', 'ADDR_REPEAT', 'ADDR_REPEAT_2',
    'INFO_CAPTURED', 'ALT_OFFERED', 'NO_SLOT', 'SLOT_OK', 'BOOK_ERROR',
    'URGENT*STANDARD_VISIT', 'URGENT*EMERGENCY_ROUTE', 'MILD_CUE*EMERGENCY_ROUTE', 'SHOPPING*SLOT_OK',
  ];
  function featurize(state) {
    const x = new Array(FEATURES.length).fill(0);
    x[0] = 1;
    for (let i = 1; i < FEATURES.length; i++) {
      const f = FEATURES[i];
      if (f === 'ADDR_REPEAT_2') x[i] = (state.ADDR_REPEAT || 0) >= 2 ? 1 : 0;
      else if (f.includes('*')) { const [a, b] = f.split('*'); x[i] = state[a] && state[b] ? 1 : 0; }
      else x[i] = state[f] ? 1 : 0;
    }
    return x;
  }
  // Walk a transcript; return prefix states after every turn (index aligned
  // with turns) plus tags per turn.
  function prefixStates(turns, cust) {
    const state = {};
    if (cust) state['cust:' + cust] = 1;
    const out = [];
    for (const [spk, text] of turns) {
      const tags = tagTurn(spk, text);
      for (const t of tags) {
        if (t === 'ADDR_REPEAT') state[t] = (state[t] || 0) + (text.match(/(repeat|spell)/gi) || []).length;
        else state[t] = 1;
      }
      out.push({ tags, state: Object.assign({}, state) });
    }
    return out;
  }

  // L2-regularised logistic regression by IRLS (Newton). X: rows of features.
  function fitLogistic(X, y, l2 = 1.0, iters = 25) {
    const d = X[0].length;
    let w = new Array(d).fill(0);
    for (let it = 0; it < iters; it++) {
      const H = Array.from({ length: d }, () => new Array(d).fill(0));
      const g = new Array(d).fill(0);
      for (let i = 0; i < X.length; i++) {
        const xi = X[i];
        let z = 0;
        for (let j = 0; j < d; j++) z += w[j] * xi[j];
        const p = sigmoid(z);
        const r = p - y[i];
        const s = p * (1 - p);
        for (let j = 0; j < d; j++) {
          if (xi[j] === 0) continue;
          g[j] += r * xi[j];
          for (let k = 0; k < d; k++) if (xi[k] !== 0) H[j][k] += s * xi[j] * xi[k];
        }
      }
      for (let j = 1; j < d; j++) { g[j] += l2 * w[j]; H[j][j] += l2; }
      H[0][0] += 1e-6;
      const step = solve(H, g);
      let maxStep = 0;
      w = w.map((v, j) => { maxStep = Math.max(maxStep, Math.abs(step[j])); return v - step[j]; });
      if (maxStep < 1e-7) break;
    }
    return w;
  }
  function solve(A, b) { // Gaussian elimination with partial pivoting
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      const v = M[c][c] || 1e-12;
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / v;
        if (f === 0) continue;
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((row, i) => row[n] / (row[i] || 1e-12));
  }
  const predict = (w, x) => sigmoid(x.reduce((s, v, j) => s + v * w[j], 0));

  // Train V on every prefix of every lead (Monte-Carlo value targets).
  function trainValueModel(calls, l2 = 1.0) {
    const X = [], y = [];
    for (const c of calls) {
      const ps = prefixStates(c.turns, c.cust);
      for (let i = 1; i < ps.length; i++) { // from the caller's opening onward
        X.push(featurize(ps[i].state));
        y.push(c.booked ? 1 : 0);
      }
    }
    return { w: fitLogistic(X, y, l2), rows: X.length };
  }

  // ------------------------------------------------------------------ 3. blame localization
  const MODE_BY_ACT = {
    EMERGENCY_ROUTE: 'EMERGENCY_FEE_DECLINED',
    STANDARD_VISIT: 'EMERGENCY_TRIAGED_ROUTINE',
    PRICE_DEFLECT: 'PRICE_DEFLECTED',
    PRICE_QUOTE: 'FEE_DECLINED',
    ADDR_REPEAT: 'ADDRESS_CAPTURE_LOOP',
    NO_SLOT: 'NO_ALTERNATIVE_SLOT',
    ALT_OFFERED: 'ALTERNATIVE_DECLINED',
    BOOK_ERROR: 'BOOKING_TOOL_ERROR',
  };
  // most damaging first when a turn carries several acts
  const RISK_ACTS = ['STANDARD_VISIT', 'NO_SLOT', 'PRICE_DEFLECT', 'ADDR_REPEAT', 'BOOK_ERROR', 'EMERGENCY_ROUTE', 'PRICE_QUOTE', 'ALT_OFFERED'];
  const MODES = {
    NO_ALTERNATIVE_SLOT: { label: 'Requested window full, no alternative offered', addressable: true },
    PRICE_DEFLECTED: { label: 'Price question deflected to technician', addressable: true },
    EMERGENCY_TRIAGED_ROUTINE: { label: 'Urgent issue triaged as a routine visit', addressable: true },
    EMERGENCY_FEE_DECLINED: { label: 'Caller declined the emergency fee', addressable: true },
    FEE_DECLINED: { label: 'Caller balked at the quoted diagnostic fee', addressable: true },
    ADDRESS_CAPTURE_LOOP: { label: 'Address capture loop (repeated asks)', addressable: true },
    ALTERNATIVE_DECLINED: { label: 'Offered alternatives declined', addressable: false },
    BOOKING_TOOL_ERROR: { label: 'Booking tool error, callback promised', addressable: false },
    CALLER_SHOPPING: { label: 'Comparison shopper, no agent-side cause', addressable: false },
  };
  // For a call: V after every turn, drops at agent turns, blamed turn + mode.
  function localize(call, w, minDrop = 0.03) {
    const ps = prefixStates(call.turns, call.cust);
    const V = ps.map((p) => predict(w, featurize(p.state)));
    let best = -1, bestDrop = -Infinity;
    const drops = new Array(ps.length).fill(null);
    for (let i = 2; i < ps.length; i++) {
      if (call.turns[i][0] !== 'agent') continue;
      // V before this agent turn vs V after the caller's reaction to it
      const after = i + 1 < ps.length && call.turns[i + 1][0] === 'caller' ? i + 1 : i;
      const drop = V[i - 1] - V[after];
      drops[i] = drop;
      if (drop > bestDrop) { bestDrop = drop; best = i; }
    }
    // Credit carry: a drop at a benign turn (e.g. a generic "let me call you
    // back" after the address is taken) is evidence about an earlier problem.
    // It is credited to the open grievance (latest risky turn the caller
    // reacted badly to), else the latest risky turn, else the turn itself.
    const credit = new Array(ps.length).fill(0);
    let open = null, lastRisky = null;
    for (let i = 2; i < ps.length; i++) {
      if (drops[i] === null) continue;
      const risky = RISK_ACTS.some((t) => ps[i].tags.includes(t));
      const reaction = i + 1 < ps.length && call.turns[i + 1][0] === 'caller' ? ps[i + 1].tags : [];
      const negative = reaction.includes('HESITANT') || reaction.includes('EXIT');
      if (risky) {
        credit[i] += drops[i];
        lastRisky = i;
        if (negative) open = i;
      } else {
        const shopper = ps[i].state.SHOPPING && open === null;
        const target = shopper ? i : (open !== null ? open : (lastRisky !== null ? lastRisky : i));
        credit[target] += drops[i];
      }
    }
    best = -1; bestDrop = -Infinity;
    for (let i = 0; i < credit.length; i++) if (drops[i] !== null && credit[i] > bestDrop) { bestDrop = credit[i]; best = i; }
    let mode = null;
    if (best >= 0 && bestDrop >= minDrop) {
      const tags = ps[best].tags;
      for (const t of RISK_ACTS) if (tags.includes(t)) { mode = MODE_BY_ACT[t]; break; }
    }
    if (!mode) { mode = ps.some((p) => p.tags.includes('SHOPPING')) ? 'CALLER_SHOPPING' : 'UNEXPLAINED'; best = -1; }
    return { V, drops, credit, blameTurn: best, drop: bestDrop, mode };
  }
  // Naive baseline: blame the last agent turn before the caller's exit.
  function lastAgentTurnBefore(call) {
    for (let i = call.turns.length - 3; i >= 0; i--) if (call.turns[i][0] === 'agent') return i;
    return -1;
  }

  // Cluster lost leads by localized mode, with job value at stake.
  function failureModes(lostResults) {
    const by = {};
    for (const r of lostResults) {
      const m = r.mode;
      by[m] = by[m] || { mode: m, calls: [], value: 0, drops: [] };
      by[m].calls.push(r.call);
      by[m].value += r.call.value;
      by[m].drops.push(r.drop);
    }
    const total = lostResults.length;
    return Object.values(by)
      .map((g) => ({
        mode: g.mode,
        label: (MODES[g.mode] || { label: 'Unexplained' }).label,
        addressable: (MODES[g.mode] || {}).addressable || false,
        count: g.calls.length,
        share: g.calls.length / total,
        value: g.value,
        medianDrop: g.drops.slice().sort((a, b) => a - b)[Math.floor(g.drops.length / 2)],
        calls: g.calls,
      }))
      .sort((a, b) => b.value - a.value);
  }

  // ------------------------------------------------------------------ 4. replay lift estimation
  // Goal: the booking rate theta the leads would have had under a proposed
  // change, estimated from a replay sample before anything goes live.
  //
  // Design: stratify leads (booked + one stratum per localized failure mode),
  // replay a sample per stratum. A replay that takes the same dialogue acts as
  // the original call ("not diverged") keeps its observed outcome. Diverged
  // replays get an LLM-judge score f on every call and a human verdict y on a
  // random subset. Per stratum, the diverged part is estimated three ways:
  //   judge-only : mean(f)                        precise, biased by the judge
  //   human-only : mean(y on labeled)             unbiased, wide
  //   PPI++      : lam*mean(f_unlabeled) + mean(y - lam*f)_labeled
  //                (prediction-powered inference, Angelopoulos et al. 2023,
  //                power-tuned lam): unbiased for any judge, tighter than
  //                human-only when the judge is informative.
  // Ops = { judge(i), human(i), diverged(i), observed(i) } over lead indices.

  // Two-phase allocation: a proportional pilot, then Neyman allocation of the
  // remaining replay budget using each stratum's pilot divergence rate.
  function designSample(strata, budget, labelBudget, seed, ops) {
    const rnd = mulberry32(seed);
    const keys = Object.keys(strata);
    const N = keys.reduce((s, k) => s + strata[k].length, 0);
    budget = Math.min(budget, N);
    const order = {}, n = {};
    keys.forEach((k) => { order[k] = shuffle(strata[k], rnd); });
    const pilot = Math.round(budget * 0.3);
    keys.forEach((k) => { n[k] = Math.min(strata[k].length, Math.max(8, Math.round(pilot * strata[k].length / N))); });
    const used = keys.reduce((s, k) => s + n[k], 0);
    const sd = {};
    keys.forEach((k) => {
      const rows = order[k].slice(0, n[k]);
      const dv = mean(rows.map((i) => (ops.diverged(i) ? 1 : 0)));
      sd[k] = Math.sqrt(Math.max(dv, 0.02) * 0.25); // worst-case Bernoulli s.d. on the diverged share
    });
    const rest = budget - used;
    if (rest > 0) {
      let tot = 0; const wt = {};
      keys.forEach((k) => { wt[k] = strata[k].length * sd[k]; tot += wt[k]; });
      keys.forEach((k) => { n[k] += Math.min(strata[k].length - n[k], Math.floor(rest * wt[k] / tot)); });
    }
    // human labels go only to diverged replays, proportional to their count
    const div = {}; let totalDiv = 0;
    keys.forEach((k) => { div[k] = order[k].slice(0, n[k]).filter((i) => ops.diverged(i)); totalDiv += div[k].length; });
    const sample = {};
    keys.forEach((k) => {
      const d = div[k];
      const m = d.length === 0 ? 0 : Math.min(d.length, Math.max(Math.min(8, d.length), Math.round(labelBudget * d.length / Math.max(1, totalDiv))));
      const rows = order[k].slice(0, n[k]);
      sample[k] = {
        N: strata[k].length,
        same: rows.filter((i) => !ops.diverged(i)),
        labeled: d.slice(0, m),
        unlabeled: d.slice(m),
      };
    });
    return sample;
  }

  function ppiMean(fl, yl, fu) {
    const nL = fl.length, nU = fu.length;
    const fall = fl.concat(fu);
    let lam = 0;
    const vf = variance(fall);
    if (vf > 1e-9 && nU > 0 && nL > 1) lam = covariance(yl, fl) / ((1 + nL / nU) * vf);
    lam = Math.max(0, Math.min(1, lam));
    const rect = yl.map((y, i) => y - lam * fl[i]);
    const est = (nU > 0 ? lam * mean(fu) : lam * mean(fl)) + mean(rect);
    const pS = (yl.reduce((s, v) => s + v, 0) + 1) / (nL + 2);
    const vr = Math.max(variance(rect), 0.25 * pS * (1 - pS));
    const v = (nU > 1 ? lam * lam * variance(fu) / nU : 0) + vr / nL;
    return { est: Math.max(0, Math.min(1, est)), v, lam };
  }

  function estimate(sample, ops, alpha = 0.05) {
    const z = normInv(1 - alpha / 2);
    const keys = Object.keys(sample);
    const N = keys.reduce((s, k) => s + sample[k].N, 0);
    const out = { judge: { est: 0, v: 0 }, human: { est: 0, v: 0 }, ppi: { est: 0, v: 0 }, strata: [] };
    for (const k of keys) {
      const S = sample[k];
      const W = S.N / N;
      const nS = S.same.length, nL = S.labeled.length, nU = S.unlabeled.length, nD = nL + nU;
      const n = nS + nD;
      const ySame = S.same.reduce((s, i) => s + ops.observed(i), 0);
      const fl = S.labeled.map(ops.judge), yl = S.labeled.map(ops.human), fu = S.unlabeled.map(ops.judge);
      const parts = {};
      if (nD === 0) {
        parts.judge = parts.human = parts.ppi = { est: 0, v: 0 };
      } else {
        const fall = fl.concat(fu);
        parts.judge = { est: mean(fall), v: variance(fall) / nD };
        const pS = (yl.reduce((s, v) => s + v, 0) + 1) / (nL + 2);
        parts.human = { est: mean(yl), v: Math.max(variance(yl), pS * (1 - pS)) / nL };
        parts.ppi = ppiMean(fl, yl, fu);
      }
      const fpc = Math.max(0, 1 - n / S.N);
      const row = { key: k, N: S.N, W, replays: n, diverged: nD, labels: nL };
      for (const m of ['judge', 'human', 'ppi']) {
        const th = (ySame + nD * parts[m].est) / n;
        // sampling variance of the stratum mean + estimation variance of the diverged part
        const v = (th * (1 - th) / n) * fpc + Math.pow(nD / n, 2) * parts[m].v;
        out[m].est += W * th; out[m].v += W * W * v;
        row[m] = th; row[m + 'Se'] = Math.sqrt(v);
      }
      row.lambda = parts.ppi.lam || 0;
      row.judgeDiv = parts.judge.est; row.humanDiv = parts.human.est;
      out.strata.push(row);
    }
    for (const m of ['judge', 'human', 'ppi']) {
      const se = Math.sqrt(out[m].v);
      out[m].se = se; out[m].lo = out[m].est - z * se; out[m].hi = out[m].est + z * se;
    }
    return out;
  }

  // ------------------------------------------------------------------ 5. A/B planning
  // Calls per arm for a two-sided two-proportion z-test.
  function abSampleSize(p1, p2, alpha = 0.05, power = 0.8) {
    const d = Math.abs(p2 - p1);
    if (d < 1e-9) return Infinity;
    const za = normInv(1 - alpha / 2), zb = normInv(power);
    const pb = (p1 + p2) / 2;
    const n = Math.pow(za * Math.sqrt(2 * pb * (1 - pb)) + zb * Math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)), 2) / (d * d);
    return Math.ceil(n);
  }

  // Monte-Carlo check: how often does each method's 95% CI cover the true
  // (simulation-only) post-change booking rate?
  function coverage(strata, budget, labelBudget, ops, truth, reps = 200, seed0 = 1) {
    const res = { judge: { hit: 0, width: 0 }, human: { hit: 0, width: 0 }, ppi: { hit: 0, width: 0 } };
    for (let r = 0; r < reps; r++) {
      const s = designSample(strata, budget, labelBudget, seed0 + r * 7919, ops);
      const e = estimate(s, ops);
      for (const m of ['judge', 'human', 'ppi']) {
        if (e[m].lo <= truth && truth <= e[m].hi) res[m].hit++;
        res[m].width += e[m].hi - e[m].lo;
      }
    }
    for (const m of Object.keys(res)) { res[m].coverage = res[m].hit / reps; res[m].width /= reps; }
    return res;
  }

  const api = {
    mulberry32, sigmoid, mean, variance, covariance, normInv, shuffle,
    tagTurn, prefixStates, featurize, FEATURES, fitLogistic, predict, trainValueModel,
    localize, lastAgentTurnBefore, failureModes, MODES,
    designSample, ppiMean, estimate, abSampleSize, coverage,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Forensics = api;
})(typeof self !== 'undefined' ? self : this);
