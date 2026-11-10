/* UI for the Booking Forensics concept demo. All numbers come from core.js. */
(async function () {
  const F = window.Forensics;
  const $ = (id) => document.getElementById(id);
  const fmt$ = (v) => '$' + Math.round(v).toLocaleString('en-US');
  const fmtK = (v) => (v < 0 ? '−' : '') + (Math.abs(v) >= 1000 ? '$' + (Math.abs(v) / 1000).toFixed(Math.abs(v) >= 100000 ? 0 : 1) + 'k' : fmt$(Math.abs(v)));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
  const pp = (v, d = 1) => (v >= 0 ? '+' : '') + (v * 100).toFixed(d) + ' pts';
  const esc = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const [dataRes, cfRes] = await Promise.all([fetch('data/calls.json'), fetch('data/counterfactuals.json')]);
  const DATA = await dataRes.json();
  const CF = await cfRes.json();
  const calls = DATA.calls;
  const custName = Object.fromEntries(DATA.meta.customers.map((c) => [c.id, c.name]));

  // ---- fit the prefix value model once on every lead, localize every lost call
  const model = F.trainValueModel(calls);
  const loc = calls.map((c) => (c.booked ? null : F.localize(c, model.w)));

  const S = { team: 'all', mode: null, call: null, cand: CF.candidates[0].id, budget: 400, labels: 60, seed: 42, ex: 0 };

  // team selector
  const teamSel = $('team');
  teamSel.innerHTML = '<option value="all">All teams</option>' + DATA.meta.customers.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  teamSel.onchange = () => { S.team = teamSel.value; S.mode = null; S.call = null; S.ex = 0; renderAll(); };

  const idxOf = () => calls.map((c, i) => i).filter((i) => S.team === 'all' || calls[i].cust === S.team);

  function renderAll() {
    const idx = idxOf();
    $('wsName').textContent = S.team === 'all' ? 'All teams' : custName[S.team];
    $('wsDot').textContent = S.team === 'all' ? 'A' : custName[S.team][0];
    const lostIdx = idx.filter((i) => !calls[i].booked);
    const results = lostIdx.map((i) => ({ i, call: calls[i], ...loc[i] }));
    const modes = F.failureModes(results);
    if (!S.mode || !modes.find((m) => m.mode === S.mode)) S.mode = modes[0].mode;
    renderKpis(idx, results, modes);
    renderModes(modes);
    renderValidation(results);
    renderCalls(results);
    renderReplay(idx, results);
  }

  // ---------------------------------------------------------------- KPIs
  function renderKpis(idx, results, modes) {
    const n = idx.length, booked = idx.filter((i) => calls[i].booked).length;
    const lostVal = results.reduce((s, r) => s + r.call.value, 0);
    const fixVal = modes.filter((m) => m.addressable).reduce((s, m) => s + m.value, 0);
    const localized = results.filter((r) => r.blameTurn >= 0);
    const agentSide = localized.length;
    const drops = localized.map((r) => r.drop).sort((a, b) => a - b);
    const k = [
      ['Leads', n.toLocaleString(), `${(n / 30).toFixed(0)} per day, Oct 1–30`],
      ['Lead booking rate', pct(booked / n), `${booked.toLocaleString()} booked`],
      ['Lost leads', (n - booked).toLocaleString(), `${agentSide} traced to an agent turn`],
      ['Job value at stake', fmtK(lostVal), `${fmtK(fixVal)} in config-fixable modes`],
      ['Median blamed-turn drop', drops.length ? (drops[Math.floor(drops.length / 2)] * 100).toFixed(0) + ' pts' : '–', 'in predicted booking probability'],
    ];
    $('kpis').innerHTML = k.map(([l, v, s]) => `<div class="card kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');
  }

  // ---------------------------------------------------------------- failure modes
  function renderModes(modes) {
    const max = Math.max(...modes.map((m) => m.count));
    $('modes').innerHTML = modes.map((m) => `
      <tr class="mode-row ${m.mode === S.mode ? 'sel' : ''}" data-mode="${m.mode}">
        <td><b style="font-weight:600">${esc(m.label)}</b></td>
        <td><span class="bar"><i style="width:${(100 * m.count / max).toFixed(1)}%"></i></span>${m.count} <span class="muted">(${pct(m.share, 0)})</span></td>
        <td class="num">${fmt$(m.value)}</td>
        <td class="num">${m.mode === 'CALLER_SHOPPING' || m.mode === 'UNEXPLAINED' ? '<span class="muted">n/a</span>' : (m.medianDrop * 100).toFixed(0) + ' pts'}</td>
        <td>${m.addressable ? '<span class="chip green">Prompt / policy</span>' : m.mode === 'BOOKING_TOOL_ERROR' ? '<span class="chip red">Integration</span>' : '<span class="chip gray">Not agent-side</span>'}</td>
      </tr>`).join('');
    document.querySelectorAll('tr.mode-row').forEach((tr) => tr.onclick = () => { S.mode = tr.dataset.mode; S.call = null; renderAll(); });
  }

  // ---------------------------------------------------------------- validation
  function renderValidation(results) {
    const agentCaused = results.filter((r) => r.call.truth.mode !== 'CALLER_SHOPPING');
    const modeAcc = results.filter((r) => r.mode === r.call.truth.mode).length / results.length;
    const turnAcc = agentCaused.filter((r) => r.blameTurn === r.call.truth.turn).length / agentCaused.length;
    const lastAcc = agentCaused.filter((r) => F.lastAgentTurnBefore(r.call) === r.call.truth.turn).length / agentCaused.length;
    const lingered = agentCaused.filter((r) => r.call.truth.lingered);
    const lingAcc = lingered.filter((r) => r.blameTurn === r.call.truth.turn).length / lingered.length;
    const meter = (v, c) => `<div class="meter"><i style="width:${(v * 100).toFixed(1)}%;background:${c}"></i></div>`;
    $('valid').innerHTML = `
      <div class="row"><div><div>Blamed turn = planted cause</div>${meter(turnAcc, '#388eff')}</div><div class="big">${pct(turnAcc)}</div></div>
      <div class="row"><div><div>Baseline: blame the last agent turn</div>${meter(lastAcc, '#94a3b8')}</div><div class="big" style="color:#47596b">${pct(lastAcc)}</div></div>
      <div class="row"><div><div>Callers who lingered before hanging up (${lingered.length})</div>${meter(lingAcc, '#6629a3')}</div><div class="big">${pct(lingAcc)}</div></div>
      <div class="row"><div>Failure mode matches planted cause</div><div class="big">${pct(modeAcc)}</div></div>
      <p class="muted" style="margin:10px 0 0">Ground truth exists only because these calls are simulated. ${Math.round(100 * lingered.length / agentCaused.length)}% of agent-caused losses come from callers who stayed on for a few turns before leaving, which is where "last turn before hang-up" fails. Value model: L2 logistic regression (IRLS) on ${model.rows.toLocaleString()} transcript prefixes.</p>`;
  }

  // ---------------------------------------------------------------- call forensics
  function renderCalls(results) {
    const list = results.filter((r) => r.mode === S.mode).sort((a, b) => b.drop - a.drop);
    const label = (F.MODES[S.mode] || { label: 'Unexplained' }).label;
    $('fxTitle').textContent = `${label} · ${list.length} calls, worst first`;
    if (!S.call || !list.find((r) => r.i === S.call)) S.call = list.length ? list[0].i : null;
    $('callList').innerHTML = list.slice(0, 80).map((r) => `
      <div class="call-item ${r.i === S.call ? 'sel' : ''}" data-i="${r.i}">
        <div class="t"><span>${r.call.id}</span>${r.blameTurn >= 0 ? `<span class="drop">−${(r.drop * 100).toFixed(0)} pts</span>` : '<span class="muted">no agent cause</span>'}</div>
        <div class="m">${esc(custName[r.call.cust])} · ${esc(r.call.svc)} · Oct ${r.call.day}</div>
      </div>`).join('') + (list.length > 80 ? `<div class="muted" style="padding:10px 16px">+${list.length - 80} more</div>` : '');
    document.querySelectorAll('.call-item').forEach((el) => el.onclick = () => { S.call = +el.dataset.i; renderCalls(results); });
    renderCallView(S.call === null ? null : results.find((r) => r.i === S.call));
  }

  function renderCallView(r) {
    if (!r) { $('callView').innerHTML = '<p class="muted">No calls in this mode.</p>'; return; }
    const c = r.call;
    const ps = F.prefixStates(c.turns, c.cust);
    const truthOk = r.call.truth.mode === r.mode;
    const truthChip = c.truth.mode === 'CALLER_SHOPPING' && r.blameTurn < 0
      ? '<span class="chip green">Matches planted cause</span>'
      : (r.blameTurn === c.truth.turn ? '<span class="chip green">Blamed turn matches planted cause</span>' : `<span class="chip amber">Planted cause: turn ${c.truth.turn} (${esc((F.MODES[c.truth.mode] || {}).label || '')})</span>`);
    // V curve
    const W = 760, H = 150, pl = 34, pr = 10, pt = 12, pb = 24;
    const xs = (i) => pl + (W - pl - pr) * (i / Math.max(1, c.turns.length - 1));
    const ys = (v) => pt + (H - pt - pb) * (1 - v);
    const path = r.V.map((v, i) => `${i ? 'L' : 'M'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join('');
    let svg = `<svg class="vchart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
    [0, 0.5, 1].forEach((g) => { svg += `<line x1="${pl}" x2="${W - pr}" y1="${ys(g)}" y2="${ys(g)}" stroke="#edf2fa"/><text x="${pl - 6}" y="${ys(g) + 4}" font-size="10" fill="#7a94b0" text-anchor="end">${g * 100}%</text>`; });
    if (r.blameTurn >= 0) svg += `<line x1="${xs(r.blameTurn)}" x2="${xs(r.blameTurn)}" y1="${pt}" y2="${H - pb}" stroke="#dc2626" stroke-dasharray="3 3"/>`;
    svg += `<path d="${path}" fill="none" stroke="#388eff" stroke-width="2.2"/>`;
    r.V.forEach((v, i) => { svg += `<circle cx="${xs(i)}" cy="${ys(v)}" r="${i === r.blameTurn ? 4.5 : 2.6}" fill="${i === r.blameTurn ? '#dc2626' : c.turns[i][0] === 'agent' ? '#388eff' : '#cbdef6'}"/>`; });
    c.turns.forEach((t, i) => { if (i % 2 === 0 || c.turns.length < 14) svg += `<text x="${xs(i)}" y="${H - 6}" font-size="10" fill="#7a94b0" text-anchor="middle">${i}</text>`; });
    svg += '</svg>';
    const turns = c.turns.map(([spk, text], i) => {
      const tags = ps[i].tags.filter((t) => t !== 'INFO_CAPTURED');
      const d = r.drops[i];
      const isBlame = i === r.blameTurn;
      const vv = spk === 'agent' && d !== null && i >= 2 ? `${(r.V[i] * 100).toFixed(0)}%<br>${isBlame ? '−' + (r.drop * 100).toFixed(0) + ' pts' : (d > 0.005 ? '−' : d < -0.005 ? '+' : '±') + Math.abs(d * 100).toFixed(0)}` : `${(r.V[i] * 100).toFixed(0)}%`;
      return `<div class="turn ${spk} ${isBlame ? 'blame' : ''}"><div class="who">${spk === 'agent' ? 'Agent' : 'Caller'} · ${i}</div><div><div class="txt">${esc(text)}</div>${tags.length ? `<div class="tags">${tags.map((t) => `<span>${t}</span>`).join('')}</div>` : ''}</div><div class="vv">${vv}</div></div>`;
    }).join('');
    $('callView').innerHTML = `
      <div class="callhead"><div><b>${c.id}</b> <span class="muted">· ${esc(custName[c.cust])} · ${esc(c.svc)} · Oct ${c.day}, 2026 · ${Math.floor(c.dur / 60)}m ${c.dur % 60}s · job value ${fmt$(c.value)}</span></div>${truthChip}</div>
      <div class="muted" style="margin-bottom:2px">Predicted booking probability after each turn. Agent turns show the change from before the turn to after the caller's reaction; credit from a later generic exit is carried back to the turn the caller first objected to.</div>
      ${svg}
      <div class="turns">${turns}</div>`;
    const bl = document.querySelector('.turn.blame');
    if (bl) bl.scrollIntoView({ block: 'nearest' });
  }

  // ---------------------------------------------------------------- replay
  function strataFor(idx) {
    const st = {};
    idx.forEach((i) => { const k = calls[i].booked ? 'booked' : 'lost:' + loc[i].mode; (st[k] = st[k] || []).push(i); });
    for (const k of Object.keys(st)) if (k !== 'booked' && st[k].length < 12) { (st['lost:OTHER'] = st['lost:OTHER'] || []).push(...st[k]); delete st[k]; }
    return st;
  }
  const opsFor = (cand) => {
    const O = CF.outcomes[cand];
    return { judge: (i) => O[i][1], human: (i) => O[i][0], diverged: (i) => O[i][2] === 1, observed: (i) => (calls[i].booked ? 1 : 0) };
  };
  const stratumLabel = (k) => k === 'booked' ? 'Booked calls (regression risk)' : k === 'lost:OTHER' ? 'Lost: other small modes' : 'Lost: ' + ((F.MODES[k.slice(5)] || { label: 'Unexplained' }).label);

  function renderReplay(idx, results) {
    // candidate cards
    const modesCount = {};
    results.forEach((r) => { modesCount[r.mode] = (modesCount[r.mode] || 0) + 1; });
    $('cands').innerHTML = CF.candidates.map((c) => `
      <div class="card cand ${c.id === S.cand ? 'sel' : ''}" data-id="${c.id}">
        <div class="n">${esc(c.name)}</div>
        <div class="d">${esc(c.change)}</div>
        <div class="f">${c.targets.map((t) => `<span class="chip blue">Targets: ${esc(F.MODES[t].label)} · ${modesCount[t] || 0} lost</span>`).join('')}</div>
      </div>`).join('');
    document.querySelectorAll('.cand').forEach((el) => el.onclick = () => { S.cand = el.dataset.id; S.ex = 0; renderReplay(idx, results); });
    $('bLbl').textContent = Math.min(S.budget, idx.length) + ' of ' + idx.length;
    $('lLbl').textContent = S.labels;

    const strata = strataFor(idx);
    const ops = opsFor(S.cand);
    const sample = F.designSample(strata, S.budget, S.labels, S.seed, ops);
    const est = F.estimate(sample, ops);
    const n = idx.length;
    const base = idx.filter((i) => calls[i].booked).length / n;
    const truth = idx.reduce((s, i) => s + CF.outcomes[S.cand][i][0], 0) / n;
    const P = est.ppi;
    const lift = P.est - base, lo = P.lo - base, hi = P.hi - base;
    const avgVal = idx.reduce((s, i) => s + calls[i].value, 0) / n;
    const usedLabels = est.strata.reduce((s, r) => s + r.labels, 0);
    const replayed = est.strata.reduce((s, r) => s + r.replays, 0);
    // recovered / regressed with intervals
    let rec = 0, recV = 0, reg = 0, regV = 0;
    est.strata.forEach((r) => {
      if (r.key === 'booked') { reg = r.N * (1 - r.ppi); regV = Math.pow(r.N * r.ppiSe, 2); }
      else { rec += r.N * r.ppi; recV += Math.pow(r.N * r.ppiSe, 2); }
    });
    const z = 1.96;
    const res = [
      ['Projected booking rate', pct(P.est), `95% CI ${pct(P.lo)} – ${pct(P.hi)} · today ${pct(base)}`],
      ['Lift', pp(lift), `95% CI ${pp(lo)} to ${pp(hi)}`],
      ['Lost leads recovered', Math.round(rec).toLocaleString(), `± ${Math.round(z * Math.sqrt(recV))} of ${est.strata.filter((r) => r.key !== 'booked').reduce((s, r) => s + r.N, 0)} lost`],
      ['Booked calls broken', Math.round(reg).toLocaleString(), `± ${Math.round(z * Math.sqrt(regV))} of ${est.strata.find((r) => r.key === 'booked').N} booked`],
      ['Booked job value / month', (lift >= 0 ? '+' : '') + fmtK(lift * n * avgVal), `95% CI ${fmtK(lo * n * avgVal)} to ${fmtK(hi * n * avgVal)}`],
    ];
    $('results').innerHTML = res.map(([l, v, s]) => `<div class="card kpi"><div class="l">${l}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join('');

    // verdict
    let vcls, vtxt;
    const half = (P.hi - P.lo) / 2;
    if (lo > 0) { vcls = 'go'; vtxt = `<b>Worth an A/B test.</b> The whole 95% interval is above today's rate. ${replayed} replays and ${usedLabels} human reviews were enough; the judge alone would have said ${pp(est.judge.est - base)}.`; }
    else if (hi < 0) { vcls = 'stop'; vtxt = `<b>Likely a regression.</b> The change breaks more booked calls than it recovers. Do not ship as written.`; }
    else {
      const need = Math.min(2000, Math.ceil(usedLabels * Math.pow(half / Math.max(Math.abs(lift), 0.002), 2)));
      const J = est.judge;
      const judgeSays = J.lo - base > 0 ? ` The LLM judge alone would have called it a clear win (${pp(J.est - base)}, CI ${pp(J.lo - base)} to ${pp(J.hi - base)}).` : '';
      vcls = 'hold'; vtxt = `<b>Inconclusive.</b> The interval straddles zero${reg > 0.25 * rec ? ': recovered lost calls are offset by booked calls the change breaks' : ''}.${judgeSays} Roughly ${need.toLocaleString()} reviewed replays would resolve a lift this small, or rework the change.`;
    }
    $('verdict').innerHTML = `<div class="verdict ${vcls}">${vtxt}</div>`;

    renderForest(est, base, truth);
    renderAB(base, P, n, lift, lo);
    $('strata').innerHTML = est.strata.sort((a, b) => b.N - a.N).map((r) => `
      <tr><td>${esc(stratumLabel(r.key))}</td><td class="num">${r.N}</td><td class="num">${r.replays}</td><td class="num">${r.diverged}</td><td class="num">${r.labels}</td>
      <td class="num">${r.diverged ? pct(r.judgeDiv) : '–'}</td><td class="num">${r.labels ? pct(r.humanDiv) : '–'}</td><td class="num">${r.diverged ? r.lambda.toFixed(2) : '–'}</td><td class="num"><b>${pct(r.ppi)}</b></td></tr>`).join('');
    renderExample(sample, ops);
    $('covCard').hidden = true;
    $('covBtn').onclick = () => runCoverage(strata, ops, truth);
  }

  function renderForest(est, base, truth) {
    const rows = [['Judge only', est.judge, '#94a3b8'], ['Human review only', est.human, '#f59e0b'], ['Judge + human (PPI++)', est.ppi, '#388eff']];
    const vals = rows.flatMap(([, e]) => [e.lo, e.hi]).concat([base, truth]);
    let mn = Math.min(...vals), mx = Math.max(...vals);
    const pad = (mx - mn) * 0.12 + 0.003; mn -= pad; mx += pad;
    const W = 640, H = 210, pl = 160, pr = 20, pt = 18;
    const x = (v) => pl + (W - pl - pr) * (v - mn) / (mx - mn);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="forest">`;
    const ticks = 5;
    for (let t = 0; t <= ticks; t++) {
      const v = mn + (mx - mn) * t / ticks;
      s += `<line x1="${x(v)}" x2="${x(v)}" y1="${pt}" y2="${H - 34}" stroke="#edf2fa"/><text x="${x(v)}" y="${H - 18}" font-size="11" fill="#7a94b0" text-anchor="middle">${(v * 100).toFixed(1)}%</text>`;
    }
    s += `<line x1="${x(base)}" x2="${x(base)}" y1="${pt - 6}" y2="${H - 34}" stroke="#0d1421" stroke-dasharray="4 3"/><text x="${x(base)}" y="${pt - 8}" font-size="11" fill="#0d1421" text-anchor="middle">today ${pct(base)}</text>`;
    s += `<line x1="${x(truth)}" x2="${x(truth)}" y1="${pt}" y2="${H - 34}" stroke="#16a34a" stroke-width="2"/>`;
    rows.forEach(([name, e, col], k) => {
      const y = pt + 26 + k * 46;
      s += `<text x="0" y="${y + 4}" font-size="12.5" fill="#3d5068">${name}</text>`;
      s += `<line x1="${x(e.lo)}" x2="${x(e.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="6" stroke-linecap="round" opacity=".35"/>`;
      s += `<circle cx="${x(e.est)}" cy="${y}" r="6" fill="${col}"/>`;
      s += `<text x="${x(e.est)}" y="${y - 11}" font-size="11" fill="#3d5068" text-anchor="middle">${pct(e.est)} ± ${((e.hi - e.lo) * 50).toFixed(1)}</text>`;
    });
    s += `<text x="${x(truth) + 4}" y="${H - 38}" font-size="10.5" fill="#16a34a">truth ${pct(truth)}</text>`;
    s += '</svg>';
    $('forest').outerHTML = s.replace('<svg ', '<svg id="forest" ');
  }

  function renderAB(base, P, n, lift, lo) {
    const perDay = n / 30;
    const nPt = lift > 0 ? F.abSampleSize(base, base + lift) : Infinity;
    const nCons = lo > 0 ? F.abSampleSize(base, base + lo) : Infinity;
    const days = (m) => (isFinite(m) ? Math.ceil(2 * m / perDay) : null);
    const line = (label, m) => `<div class="row" style="display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #edf2fa"><span>${label}</span><span><b>${isFinite(m) ? m.toLocaleString() : '–'}</b> leads / arm${isFinite(m) ? ` <span class="muted">· ~${days(m)} days at ${perDay.toFixed(0)} leads/day</span>` : ''}</span></div>`;
    $('abplan').innerHTML = `
      <div class="muted" style="margin-bottom:6px">Two-sided test, α = 0.05, 80% power, 50/50 split, lead booking rate as the primary metric.</div>
      ${line('Size for the point estimate', nPt)}
      ${line('Size for the lower 95% bound', nCons)}
      <p class="muted" style="margin:10px 0 12px">${lo > 0 ? 'Sizing on the lower bound means the experiment is powered even if the true lift is at the pessimistic end of the offline estimate.' : 'The offline interval includes zero, so there is no defensible effect size to power a live test for yet.'} If you plan to check results weekly, use a sequential or Bayesian design instead of a fixed horizon.</p>
      <button class="btn primary" disabled title="Concept: would prefill Analytics → A/B Test with the two configs, hypothesis and duration">Create A/B experiment</button>`;
  }

  function runCoverage(strata, ops, truth) {
    $('covCard').hidden = false;
    $('cov').innerHTML = '<p class="muted">Running 200 resampled replay designs…</p>';
    setTimeout(() => {
      const t0 = performance.now();
      const cov = F.coverage(strata, S.budget, S.labels, ops, truth, 200, 101);
      const ms = performance.now() - t0;
      const row = (name, c, col) => `
        <div style="display:grid;grid-template-columns:200px 1fr 90px 120px;gap:12px;align-items:center;padding:8px 0;border-bottom:1px solid #edf2fa">
          <span>${name}</span>
          <div class="meter" style="margin:0;position:relative"><i style="width:${(c.coverage * 100).toFixed(1)}%;background:${col}"></i><span style="position:absolute;left:95%;top:-3px;height:16px;border-left:2px dashed #0d1421"></span></div>
          <b>${pct(c.coverage, 0)}</b><span class="muted">avg width ${(c.width * 100).toFixed(1)} pts</span>
        </div>`;
      $('cov').innerHTML = row('Judge only', cov.judge, '#94a3b8') + row('Human review only', cov.human, '#f59e0b') + row('Judge + human (PPI++)', cov.ppi, '#388eff') +
        `<p class="muted" style="margin:10px 0 0">Share of 95% intervals that contain the simulation's true post-change booking rate (${pct(truth)}). Dashed line = 95% target. A judge-only interval can be narrow and still wrong; PPI++ keeps coverage near nominal while staying narrower than review alone. Computed in ${ms.toFixed(0)} ms.</p>`;
      $('covHint').textContent = `200 resamples · ${Math.min(S.budget, idxOf().length)} replays · ${S.labels} reviews`;
    }, 30);
  }

  function renderExample(sample, ops) {
    const cand = CF.candidates.find((c) => c.id === S.cand);
    const pool = [];
    Object.entries(sample).forEach(([k, s]) => {
      s.labeled.forEach((i) => pool.push({ i, k, labeled: true }));
      s.unlabeled.forEach((i) => pool.push({ i, k, labeled: false }));
    });
    // put recovered lost calls first, then regressions, then the rest
    const rank = (p) => (p.k !== 'booked' && CF.outcomes[S.cand][p.i][0] === 1 ? 0 : p.k === 'booked' && CF.outcomes[S.cand][p.i][0] === 0 ? 1 : 2);
    pool.sort((a, b) => rank(a) - rank(b) || a.i - b.i);
    if (!pool.length) { $('example').innerHTML = '<p class="muted">No diverged replays in this sample.</p>'; return; }
    const p = pool[S.ex % pool.length];
    const c = calls[p.i];
    const rep = CF.replays[S.cand][c.id];
    const from = rep.from;
    const orig = c.turns.slice(Math.max(0, from - 1));
    const repl = (from > 0 ? [c.turns[from - 1]] : []).concat(rep.turns);
    const ln = (t, cls) => `<div class="ln ${cls}"><b>${t[0]}</b>${esc(t[1])}</div>`;
    const f = ops.judge(p.i), y = ops.human(p.i);
    $('exHint').innerHTML = `${pool.length} diverged replays in this sample · <a href="#" id="nextEx">next example</a>`;
    $('example').innerHTML = `
      <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap">
        <b>${c.id}</b><span class="muted">${esc(custName[c.cust])} · ${esc(c.svc)} · ${esc(stratumLabel(p.k))}</span>
        <span class="chip ${c.booked ? 'green' : 'red'}">Original: ${c.booked ? 'booked' : 'lost'}</span>
        <span class="chip violet">Judge: ${pct(f, 0)} likely to book</span>
        ${p.labeled ? `<span class="chip ${y ? 'green' : 'red'}">Human review: ${y ? 'would book' : 'would not book'}</span>` : '<span class="chip gray">Not reviewed (judge only)</span>'}
      </div>
      <div class="diff">
        <div class="col"><h4>Original call, from turn ${Math.max(0, from - 1)}</h4>${orig.map((t, k) => ln(t, k === (from > 0 ? 1 : 0) ? 'old' : '')).join('')}</div>
        <div class="col"><h4>Replay with “${esc(cand.short)}”</h4>${repl.map((t, k) => ln(t, k === (from > 0 ? 1 : 0) ? 'new' : '')).join('')}</div>
      </div>`;
    $('nextEx').onclick = (e) => { e.preventDefault(); S.ex++; renderExample(sample, ops); };
  }

  $('budget').oninput = (e) => { S.budget = +e.target.value; S.ex = 0; renderReplay(idxOf(), idxOf().filter((i) => !calls[i].booked).map((i) => ({ i, call: calls[i], ...loc[i] }))); };
  $('labels').oninput = (e) => { S.labels = +e.target.value; S.ex = 0; renderReplay(idxOf(), idxOf().filter((i) => !calls[i].booked).map((i) => ({ i, call: calls[i], ...loc[i] }))); };
  $('resample').onclick = () => { S.seed = (S.seed * 48271) % 2147483647; S.ex = 0; renderReplay(idxOf(), idxOf().filter((i) => !calls[i].booked).map((i) => ({ i, call: calls[i], ...loc[i] }))); };

  renderAll();
})();
