/* UI for the Booking Forensics concept, styled as a Responder tab. All numbers come from core.js. */
(async function () {
  const F = window.Forensics;
  const $ = (id) => document.getElementById(id);
  const icons = () => { if (window.lucide) window.lucide.createIcons(); };
  const I = (n) => `<i data-lucide="${n}"></i>`;
  const fmt$ = (v) => '$' + Math.round(v).toLocaleString('en-US');
  const fmtK = (v) => (v < 0 ? '−' : '') + (Math.abs(v) >= 1000 ? '$' + (Math.abs(v) / 1000).toFixed(Math.abs(v) >= 100000 ? 0 : 1) + 'k' : fmt$(Math.abs(v)));
  const pct = (v, d = 1) => (v * 100).toFixed(d) + '%';
  const pp = (v, d = 1) => (v >= 0 ? '+' : '−') + Math.abs(v * 100).toFixed(d) + ' pts';
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  const [dataRes, cfRes] = await Promise.all([fetch('data/calls.json'), fetch('data/counterfactuals.json')]);
  const DATA = await dataRes.json();
  const CF = await cfRes.json();
  const calls = DATA.calls;
  const custName = Object.fromEntries(DATA.meta.customers.map((c) => [c.id, c.name]));

  const model = F.trainValueModel(calls);
  const loc = calls.map((c) => (c.booked ? null : F.localize(c, model.w)));
  const S = { team: 'all', mode: null, call: null, cand: CF.candidates[0].id, budget: 400, labels: 60, seed: 42, ex: 0 };

  const teamSel = $('team');
  teamSel.innerHTML = '<option value="all">All teams</option>' + DATA.meta.customers.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
  teamSel.onchange = () => { S.team = teamSel.value; S.mode = null; S.call = null; S.ex = 0; renderAll(); };
  const idxOf = () => calls.map((c, i) => i).filter((i) => S.team === 'all' || calls[i].cust === S.team);
  const lostResults = (idx) => idx.filter((i) => !calls[i].booked).map((i) => ({ i, call: calls[i], ...loc[i] }));

  // smooth path through points (Catmull-Rom -> cubic Bezier), the curve style used by Avoca's charts
  function smooth(pts, yMin = -1e9, yMax = 1e9) {
    const cl = (v) => Math.max(yMin, Math.min(yMax, v));
    if (pts.length < 2) return '';
    let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      const t = 0.18;
      const c1 = [p1[0] + (p2[0] - p0[0]) * t, cl(p1[1] + (p2[1] - p0[1]) * t)];
      const c2 = [p2[0] - (p3[0] - p1[0]) * t, cl(p2[1] - (p3[1] - p1[1]) * t)];
      d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
    }
    return d;
  }
  const fixFor = (mode) => CF.candidates.find((c) => c.targets.includes(mode));
  const clock = (c) => { const m = 8 * 60 + (parseInt(c.id.slice(4), 10) * 37) % 600; const h = Math.floor(m / 60); return `${((h + 11) % 12) + 1}:${String(m % 60).padStart(2, '0')} ${h < 12 ? 'AM' : 'PM'}`; };
  const kpi = (icon, color, label, value, sub) => `<div class="card kpi"><div class="ic ${color}">${I(icon)}</div><div style="min-width:0"><div class="l">${label}</div><div class="v">${value}</div><div class="d">${sub}</div></div></div>`;

  function renderAll() {
    const idx = idxOf();
    $('teamNm').textContent = S.team === 'all' ? 'All teams' : custName[S.team];
    $('teamAv').textContent = S.team === 'all' ? 'A' : custName[S.team][0];
    const results = lostResults(idx);
    const modes = F.failureModes(results);
    if (!S.mode || !modes.find((m) => m.mode === S.mode)) S.mode = modes[0].mode;
    renderKpis(idx, results, modes);
    renderModes(modes);
    renderTrust(results);
    renderCalls(results);
    renderReplay(idx, results);
  }

  function renderKpis(idx, results, modes) {
    const n = idx.length, booked = idx.filter((i) => calls[i].booked).length;
    const lostVal = results.reduce((s, r) => s + r.call.value, 0);
    const fixVal = modes.filter((m) => m.addressable).reduce((s, m) => s + m.value, 0);
    const traced = results.filter((r) => r.blameTurn >= 0).length;
    $('kpis').innerHTML = [
      kpi('phone', 'purple', 'Leads', n.toLocaleString(), `${(n / 30).toFixed(0)} per day`),
      kpi('percent', 'green', `Booking Rate ${I('info')}`, pct(booked / n), `${booked.toLocaleString()} booked`),
      kpi('phone-missed', 'red', 'Lost Leads', (n - booked).toLocaleString(), `${pct((n - booked) / n)} of leads`),
      kpi('dollar-sign', 'yellow', 'Job Value at Stake', fmtK(lostVal), `${fmtK(fixVal)} fixable in config`),
      kpi('crosshair', 'blue', `Traced to a Turn ${I('info')}`, pct(traced / results.length), `${traced} of ${results.length} lost leads`),
    ].join('');
  }

  function renderModes(modes) {
    $('modes').innerHTML = modes.map((m) => {
      const fix = m.addressable ? '<span class="pill green">Prompt / policy</span>' : m.mode === 'BOOKING_TOOL_ERROR' ? '<span class="pill red">Integration</span>' : '<span class="pill gray">Not agent-side</span>';
      const drop = m.mode === 'CALLER_SHOPPING' || m.mode === 'UNEXPLAINED' ? '<span class="muted">n/a</span>' : `<span class="pill yellow">−${(m.medianDrop * 100).toFixed(0)} pts</span>`;
      return `<tr class="click ${m.mode === S.mode ? 'on' : ''}" data-mode="${m.mode}"><td class="wrap" style="white-space:normal">${esc(m.label)}</td><td class="num">${m.count} <span class="muted">· ${pct(m.share, 0)}</span></td><td class="num">${fmt$(m.value)}</td><td class="num">${drop}</td><td>${fix}</td></tr>`;
    }).join('');
    document.querySelectorAll('#modes tr').forEach((tr) => tr.onclick = () => { S.mode = tr.dataset.mode; S.call = null; renderAll(); });
  }

  function renderTrust(results) {
    const agentCaused = results.filter((r) => r.call.truth.mode !== 'CALLER_SHOPPING');
    const modeAcc = results.filter((r) => r.mode === r.call.truth.mode).length / results.length;
    const turnAcc = agentCaused.filter((r) => r.blameTurn === r.call.truth.turn).length / agentCaused.length;
    const lastAcc = agentCaused.filter((r) => F.lastAgentTurnBefore(r.call) === r.call.truth.turn).length / agentCaused.length;
    const ling = agentCaused.filter((r) => r.call.truth.lingered);
    const lingAcc = ling.filter((r) => r.blameTurn === r.call.truth.turn).length / ling.length;
    const item = (q, v, n, col) => `<div class="it" style="cursor:default"><div class="q"><span>${q}</span><b style="color:${col}">${pct(v)}</b></div><div class="bar"><i style="width:${(v * 100).toFixed(1)}%;background:${col}"></i></div><div class="c">${n}</div></div>`;
    $('trust').innerHTML = `
      <div class="t"><span style="font-size:15px;color:#2e2e2e;font-weight:600">Attribution Accuracy</span><span class="pill gray">Simulated ground truth</span></div>
      <div class="s" style="margin-top:2px">Checked against the cause planted in each synthetic call</div>
      ${item('Blamed turn matches the planted cause', turnAcc, `${agentCaused.length} agent-caused losses`, '#16a34a')}
      ${item('Baseline: blame the last agent turn', lastAcc, 'Same calls, naive rule', '#9ca3af')}
      ${item('Callers who lingered before hanging up', lingAcc, `${ling.length} calls where the naive rule fails`, '#388eff')}
      ${item('Cause category matches', modeAcc, `${results.length} lost leads`, '#16a34a')}
      <div class="muted" style="margin-top:8px">Prefix value model: L2 logistic regression (IRLS) on ${model.rows.toLocaleString()} transcript prefixes. On real calls, validate against Coach call reviews.</div>`;
  }

  function renderCalls(results) {
    const list = results.filter((r) => r.mode === S.mode).sort((a, b) => b.drop - a.drop || a.i - b.i);
    const label = (F.MODES[S.mode] || { label: 'Unexplained' }).label;
    $('fxCount').textContent = `${list.length} calls · ${label}`;
    if (S.call === null || !list.find((r) => r.i === S.call)) S.call = list.length ? list[0].i : null;
    $('callList').innerHTML = `<div class="tbl-tools"><div class="search">${I('search')}Search customers or call IDs…</div></div>` + list.slice(0, 60).map((r) => `
      <div class="ci ${r.i === S.call ? 'on' : ''}" data-i="${r.i}">
        <div class="a">${r.call.id}</div><div class="drop">${r.blameTurn >= 0 ? '−' + (r.drop * 100).toFixed(0) + ' pts' : '<span class="pill gray">Caller-side</span>'}</div>
        <div class="b">${esc(custName[r.call.cust])} · ${esc(r.call.svc)}</div><div class="b" style="grid-column:2;text-align:right">Oct ${r.call.day}</div>
      </div>`).join('') + (list.length > 60 ? `<div class="muted" style="padding:10px 14px">+${list.length - 60} more calls</div>` : '');
    document.querySelectorAll('.ci').forEach((el) => el.onclick = () => { S.call = +el.dataset.i; renderCalls(results); });
    renderCallView(S.call === null ? null : results.find((r) => r.i === S.call));
    icons();
  }

  function renderCallView(r) {
    if (!r) { $('callView').innerHTML = '<div class="card ch muted">No calls for this cause.</div>'; return; }
    const c = r.call;
    const ps = F.prefixStates(c.turns, c.cust);
    // per-turn timestamps from cumulative speaking length (synthetic)
    let t = 2; const ts = c.turns.map(([, x]) => { const s = t; t += Math.max(2, Math.round(x.length / 14)); return s; });
    const mmss = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    const W = 820, H = 190, pl = 40, pr = 14, pt = 14, pb = 26;
    const xs = (i) => pl + (W - pl - pr) * (i / Math.max(1, c.turns.length - 1));
    const ys = (v) => pt + (H - pt - pb) * (1 - v);
    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block">`;
    [0, 0.25, 0.5, 0.75, 1].forEach((g) => { svg += `<line x1="${pl}" x2="${W - pr}" y1="${ys(g)}" y2="${ys(g)}" stroke="#e5e7eb" stroke-dasharray="3 3"/><text x="${pl - 8}" y="${ys(g) + 4}" font-size="11" fill="#6b7280" text-anchor="end">${g * 100}%</text>`; });
    c.turns.forEach((_, i) => { svg += `<line x1="${xs(i)}" x2="${xs(i)}" y1="${pt}" y2="${H - pb}" stroke="#f1f3f6" stroke-dasharray="3 3"/><text x="${xs(i)}" y="${H - 8}" font-size="11" fill="#6b7280" text-anchor="middle">${mmss(ts[i])}</text>`; });
    if (r.blameTurn >= 0) svg += `<line x1="${xs(r.blameTurn)}" x2="${xs(r.blameTurn)}" y1="${pt}" y2="${H - pb}" stroke="#ef4444" stroke-dasharray="4 3"/>`;
    const pts = r.V.map((v, i) => [xs(i), ys(v)]);
    svg += `<path d="${smooth(pts, pt, H - pb)}" fill="none" stroke="#22c55e" stroke-width="2.2"/>`;
    pts.forEach((p, i) => { svg += `<circle cx="${p[0]}" cy="${p[1]}" r="${i === r.blameTurn ? 5 : 3}" fill="${i === r.blameTurn ? '#ef4444' : '#22c55e'}"/>`; });
    svg += '</svg>';
    const avgV = r.V.reduce((s, v) => s + v, 0) / r.V.length;
    const blameTags = r.blameTurn >= 0 ? ps[r.blameTurn].tags.filter((x) => x !== 'INFO_CAPTURED') : [];
    const label = (F.MODES[r.mode] || { label: 'Unexplained' }).label;
    const truthOk = r.blameTurn === c.truth.turn || (c.truth.mode === 'CALLER_SHOPPING' && r.blameTurn < 0);
    const turns = c.turns.map(([spk, text], i) => {
      const tags = ps[i].tags.filter((x) => x !== 'INFO_CAPTURED');
      const d = r.drops[i];
      const isB = i === r.blameTurn;
      const vv = `${(r.V[i] * 100).toFixed(0)}%` + (spk === 'agent' && d !== null ? `<br>${isB ? '−' + (r.drop * 100).toFixed(0) + ' pts' : (d > 0.005 ? '−' : d < -0.005 ? '+' : '±') + Math.abs(d * 100).toFixed(0)}` : '');
      return `<div class="turn ${spk} ${isB ? 'blame' : ''}"><div class="ts">${mmss(ts[i])}</div><div><div class="sp">${spk === 'agent' ? 'Avoca AI' : 'Customer'}</div><div class="tx">${esc(text)}</div>${tags.length ? `<div class="tg">${tags.map((x) => `<span>${x}</span>`).join('')}</div>` : ''}</div><div class="vv">${vv}</div></div>`;
    }).join('');
    $('callView').innerHTML = `
      <div class="cd-h"><div><span class="nm">${c.id}</span><span class="ph2">${esc(custName[c.cust])}</span></div>
        <div class="ctrls"><button class="btn">${I('copy')}Copy call link</button><button class="btn">${I('square-pen')}Edit call details</button><button class="btn">${I('external-link')}ServiceTitan</button></div></div>
      <div class="card">
        <div class="strip">
          <div><div class="k">When</div><div class="v">${I('calendar')}Oct ${c.day}, 2026 · ${clock(c)}</div></div>
          <div><div class="k">Direction</div><div class="v">${I('arrow-down-left')}Inbound</div></div>
          <div><div class="k">Duration</div><div class="v">${I('clock')}${t >= 60 ? Math.floor(t / 60) + ' min ' : ''}${t % 60} sec</div></div>
          <div><div class="k">Handled by</div><div class="v">${I('bot')}<a href="#">Avoca AI</a></div></div>
          <div><div class="k">Job value</div><div class="v">${I('wrench')}${fmt$(c.value)}</div></div>
        </div>
        <div class="chips"><span class="chip red">${I('circle-x')}Unbooked</span><span class="chip gray">${esc(c.svc)}</span>${r.blameTurn >= 0 ? `<span class="chip amber">${I('crosshair')}Lost at ${mmss(ts[r.blameTurn])} · −${(r.drop * 100).toFixed(0)} pts</span>` : '<span class="chip gray">No agent-side cause</span>'}<span class="chip ${truthOk ? 'green' : 'amber'}">${I(truthOk ? 'circle-check' : 'circle-alert')}${truthOk ? 'Matches planted cause' : 'Planted cause at ' + mmss(ts[c.truth.turn] || 0)}</span></div>
        <div class="audio"><span class="play">${I('play')}</span><span class="tm">0:00 / ${mmss(t)}</span><div class="track">${r.blameTurn >= 0 ? `<i class="mark" style="left:${(100 * ts[r.blameTurn] / t).toFixed(1)}%" title="Blamed turn"></i>` : ''}</div>${I('volume-2')}<span class="tm" style="color:#2e2e2e">1x</span>${I('ellipsis')}</div>
        <div style="padding:14px 18px 6px">
          <div style="display:flex;justify-content:space-between;align-items:flex-end"><div><div style="font-size:14px;color:#4b5563">Booking Probability</div><div style="font-size:21px;font-weight:600">${pct(r.V[Math.max(0, r.blameTurn - 1)] || r.V[0], 0)} <span style="font-size:14px;color:#dc2626;font-weight:500">${I('trending-down').replace('<i ', '<i style="width:14px;height:14px;vertical-align:-2px" ')} ${(r.drop * 100).toFixed(0)} pts at the blamed turn</span></div></div>
          <div class="legend"><span><i style="background:#22c55e"></i>Booking probability</span><span><i style="background:#ef4444"></i>Blamed turn</span></div></div>
          ${svg}
        </div>
      </div>
      ${r.blameTurn >= 0 ? `<div class="panel-amber mt"><div class="hd">${I('alarm-clock')}Lost booking: ${esc(label)}</div><div class="bd">${fixFor(r.mode) ? `<div class="sug"><div class="box"><div class="k">Current behavior</div><div><span class="dt" style="background:#9ca3af"></span>${esc(label)}</div></div>${I('arrow-right')}<div class="box y"><div class="k">Suggested change</div><div><span class="dt" style="background:#f59e0b"></span>${esc(fixFor(r.mode).short)}</div></div></div>` : ''}Predicted booking probability fell ${(r.drop * 100).toFixed(0)} points across the agent turn at ${mmss(ts[r.blameTurn])}${blameTags.length ? ` (<span class="mono" style="font-size:12px">${blameTags.join(', ')}</span>)` : ''} and the caller's reaction to it.${c.truth.lingered ? ' The caller stayed on for a few more turns before leaving, so the hang-up line is not where it went wrong.' : ''} ${(F.MODES[r.mode] || {}).addressable ? 'This cause is fixable in Responder configuration.' : ''}${fixFor(r.mode) ? `<div style="margin-top:12px;display:flex;gap:8px"><button class="btn">Dismiss</button><button class="btn primary" id="replayFix" data-cand="${fixFor(r.mode).id}">Replay this change</button></div>` : ''}</div></div>` : ''}
      <div class="card mt"><div class="ttabs"><span>Summary</span><span class="on">Transcript</span><span>Coaching</span><span>Activity</span></div><div class="turns">${turns}</div></div>`;
    const rf = $('replayFix');
    if (rf) rf.onclick = () => { S.cand = rf.dataset.cand; rerender(); document.getElementById('replaySec').scrollIntoView({ behavior: 'smooth', block: 'start' }); };
    const box = document.querySelector('#callView .turns');
    const bl = box && box.querySelector('.turn.blame');
    if (bl) box.scrollTop = Math.max(0, bl.offsetTop - 90);
  }

  // ---------------------------------------------------------------- replay
  function strataFor(idx) {
    const st = {};
    idx.forEach((i) => { const k = calls[i].booked ? 'booked' : 'lost:' + loc[i].mode; (st[k] = st[k] || []).push(i); });
    for (const k of Object.keys(st)) if (k !== 'booked' && st[k].length < 12) { (st['lost:OTHER'] = st['lost:OTHER'] || []).push(...st[k]); delete st[k]; }
    return st;
  }
  const opsFor = (cand) => { const O = CF.outcomes[cand]; return { judge: (i) => O[i][1], human: (i) => O[i][0], diverged: (i) => O[i][2] === 1, observed: (i) => (calls[i].booked ? 1 : 0) }; };
  const stratumLabel = (k) => k === 'booked' ? 'Booked calls (regression risk)' : k === 'lost:OTHER' ? 'Lost · other small causes' : 'Lost · ' + ((F.MODES[k.slice(5)] || { label: 'Unexplained' }).label);

  function renderReplay(idx, results) {
    const cand = CF.candidates.find((c) => c.id === S.cand);
    const modesCount = {};
    results.forEach((r) => { modesCount[r.mode] = (modesCount[r.mode] || 0) + 1; });
    $('cands').innerHTML = CF.candidates.map((c) => `<button class="${c.id === S.cand ? 'on' : ''}" data-id="${c.id}">${esc(c.short)}</button>`).join('');
    $('candDesc').innerHTML = `${esc(cand.change)}<div style="margin-top:8px">${cand.targets.map((t) => `<span class="pill blue">Targets: ${esc(F.MODES[t].label)} · ${modesCount[t] || 0} lost</span>`).join(' ')}</div>`;
    document.querySelectorAll('#cands button').forEach((b) => b.onclick = () => { S.cand = b.dataset.id; S.ex = 0; renderReplay(idx, results); });
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
    let rec = 0, recV = 0, reg = 0, regV = 0;
    est.strata.forEach((r) => {
      if (r.key === 'booked') { reg = r.N * (1 - r.ppi); regV = Math.pow(r.N * r.ppiSe, 2); }
      else { rec += r.N * r.ppi; recV += Math.pow(r.N * r.ppiSe, 2); }
    });
    const nLost = est.strata.filter((r) => r.key !== 'booked').reduce((s, r) => s + r.N, 0);
    const nBooked = (est.strata.find((r) => r.key === 'booked') || { N: 0 }).N;
    const trend = (v) => v >= 0 ? `<span class="up">${I('trending-up')}</span>` : `<span class="down">${I('trending-down')}</span>`;
    $('results').innerHTML = [
      kpi('percent', 'green', `Projected Rate ${I('info')}`, pct(P.est), `${trend(lift)}<span class="${lift >= 0 ? 'up' : 'down'}">${pp(lift)}</span> vs ${pct(base)}`),
      kpi('chart-line', 'blue', 'Lift, 95% CI', `${pp(lo, 1).replace(' pts', '')} to ${pp(hi, 1).replace(' pts', '')}`, 'points of booking rate'),
      kpi('rotate-ccw', 'purple', 'Leads Recovered', Math.round(rec).toLocaleString(), `± ${Math.round(1.96 * Math.sqrt(recV))} of ${nLost} lost`),
      kpi('circle-x', 'red', 'Bookings Broken', Math.round(reg).toLocaleString(), `± ${Math.round(1.96 * Math.sqrt(regV))} of ${nBooked} booked`),
      kpi('dollar-sign', 'yellow', 'Job Value / Month', (lift >= 0 ? '+' : '') + fmtK(lift * n * avgVal), `${fmtK(lo * n * avgVal)} to ${fmtK(hi * n * avgVal)}`),
    ].join('');

    let cls, head, body;
    const half = (P.hi - P.lo) / 2;
    const J = est.judge;
    if (lo > 0) { cls = 'go'; head = `${I('circle-check')}Ready for an A/B test`; body = `The whole 95% interval is above today's booking rate. ${replayed} replays and ${usedLabels} human reviews were enough; the judge alone would have reported ${pp(J.est - base)}.`; }
    else if (hi < 0) { cls = 'stop'; head = `${I('circle-x')}Likely regression`; body = 'The change breaks more booked calls than it recovers. Do not ship as written.'; }
    else {
      const need = Math.min(2000, Math.ceil(usedLabels * Math.pow(half / Math.max(Math.abs(lift), 0.002), 2)));
      cls = ''; head = `${I('alarm-clock')}Review needed (inconclusive)`;
      body = `The interval straddles zero${reg > 0.25 * rec ? ': lost calls it recovers are offset by booked calls it breaks' : ''}.${J.lo - base > 0 ? ` <b>The LLM judge alone would have called it a clear win</b> (${pp(J.est - base)}, CI ${pp(J.lo - base)} to ${pp(J.hi - base)}).` : ''} Roughly ${need.toLocaleString()} reviewed replays would resolve a lift this small, or rework the change.`;
    }
    $('verdict').innerHTML = `<div class="panel-amber verdict ${cls}"><div class="hd">${head}</div><div class="bd">${body}</div></div>`;
    $('forestBig').innerHTML = `${pct(P.est)} <span style="font-size:14px;font-weight:500;color:${lift >= 0 ? '#16a34a' : '#dc2626'}">${pp(lift)} vs today</span>`;
    renderForest(est, base, truth);
    renderAB(base, n, lift, lo);
    $('strata').innerHTML = est.strata.sort((a, b) => b.N - a.N).map((r) => `
      <tr><td>${esc(stratumLabel(r.key))}</td><td class="num">${r.N}</td><td class="num">${r.replays}</td><td class="num">${r.diverged}</td><td class="num">${r.labels}</td>
      <td class="num">${r.diverged ? pct(r.judgeDiv) : '–'}</td><td class="num">${r.labels ? pct(r.humanDiv) : '–'}</td><td class="num">${r.diverged ? r.lambda.toFixed(2) : '–'}</td><td class="num"><span class="pill ${r.ppi >= 0.5 ? 'green' : r.ppi > 0 ? 'yellow' : 'gray'}">${pct(r.ppi)}</span></td></tr>`).join('');
    renderExample(sample, ops);
    $('covCard').hidden = true;
    $('covBtn').onclick = () => runCoverage(strata, ops, truth);
    icons();
  }

  function renderForest(est, base, truth) {
    const rows = [['Judge only', est.judge, '#9ca3af'], ['Review only', est.human, '#eab308'], ['Judge + review (PPI++)', est.ppi, '#388eff']];
    const vals = rows.flatMap(([, e]) => [e.lo, e.hi]).concat([base, truth]);
    let mn = Math.min(...vals), mx = Math.max(...vals);
    const pad = (mx - mn) * 0.1 + 0.003; mn -= pad; mx += pad;
    const W = 700, H = 190, pl = 150, pr = 80, pt = 24, pb = 24;
    const x = (v) => pl + (W - pl - pr) * (v - mn) / (mx - mn);
    let s = `<svg viewBox="0 0 ${W} ${H}" class="forest" id="forest">`;
    for (let t = 0; t <= 5; t++) { const v = mn + (mx - mn) * t / 5; s += `<line x1="${x(v)}" x2="${x(v)}" y1="${pt}" y2="${H - pb}" stroke="#e5e7eb" stroke-dasharray="3 3"/><text x="${x(v)}" y="${H - 8}" font-size="11" fill="#6b7280" text-anchor="middle">${(v * 100).toFixed(1)}%</text>`; }
    s += `<line x1="${x(base)}" x2="${x(base)}" y1="${pt - 8}" y2="${H - pb}" stroke="#6b7280" stroke-dasharray="5 4"/><text x="${x(base)}" y="${pt - 12}" font-size="11" fill="#4b5563" text-anchor="middle">Today ${pct(base)}</text>`;
    s += `<line x1="${x(truth)}" x2="${x(truth)}" y1="${pt}" y2="${H - pb}" stroke="#22c55e" stroke-width="2"/>`;
    rows.forEach(([name, e, col], k) => {
      const y = pt + 26 + k * 46;
      s += `<text x="0" y="${y + 4}" font-size="13" fill="#2e2e2e">${name}</text>`;
      s += `<line x1="${x(e.lo)}" x2="${x(e.hi)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="8" stroke-linecap="round" opacity=".28"/>`;
      s += `<line x1="${x(e.lo)}" x2="${x(e.lo)}" y1="${y - 6}" y2="${y + 6}" stroke="${col}" stroke-width="2"/><line x1="${x(e.hi)}" x2="${x(e.hi)}" y1="${y - 6}" y2="${y + 6}" stroke="${col}" stroke-width="2"/>`;
      s += `<circle cx="${x(e.est)}" cy="${y}" r="5.5" fill="${col}" stroke="#fff" stroke-width="2"/>`;
      s += `<text x="${x(e.hi) + 10}" y="${y + 4}" font-size="12" fill="#4b5563">${pct(e.est)} ± ${((e.hi - e.lo) * 50).toFixed(1)}</text>`;
    });
    s += `<text x="${x(truth) + 5}" y="${H - pb - 4}" font-size="11" fill="#16a34a">truth ${pct(truth)}</text></svg>`;
    $('forest').outerHTML = s;
  }

  function renderAB(base, n, lift, lo) {
    const perDay = n / 30;
    const nPt = lift > 0 ? F.abSampleSize(base, base + lift) : Infinity;
    const nCons = lo > 0 ? F.abSampleSize(base, base + lo) : Infinity;
    const row = (label, m) => {
      const ok = isFinite(m) && lo > 0;
      return `<div class="abrow"><span>${label}</span><span style="text-align:right">${ok ? `<b>${m.toLocaleString()}</b> leads / variant <span class="muted">· ~${Math.ceil(2 * m / perDay)} days at ${perDay.toFixed(0)}/day</span>` : '<span class="muted">Not sized: interval includes zero</span>'}</span></div>`;
    };
    $('abplan').innerHTML = `
      <div class="t"><span style="font-size:15px;color:#2e2e2e;font-weight:600">A/B Test Plan</span><span class="pill gray">50/50 split</span></div>
      <div class="s" style="margin:2px 0 8px">Primary metric: Lead Booking Rate · α 0.05 · 80% power</div>
      ${row('Size for the point estimate', nPt)}
      ${row('Size for the lower 95% bound', nCons)}
      <p class="muted" style="margin:12px 0 14px">${lo > 0 ? 'Sizing on the lower bound keeps the experiment powered even if the true lift is at the pessimistic end of the offline estimate.' : 'The offline interval includes zero, so there is no defensible effect size to power a live test for yet.'} If results will be checked weekly, use a sequential or Bayesian design.</p>
      <button class="btn primary" ${lo > 0 ? '' : 'disabled'} title="Concept: would prefill Analytics → A/B Test with both assistant configs, the hypothesis and duration">${I('plus')}New Experiment</button>`;
  }

  function runCoverage(strata, ops, truth) {
    const card = $('covCard'); card.hidden = false;
    card.innerHTML = '<div class="ch muted">Running 200 resampled replay designs…</div>';
    setTimeout(() => {
      const t0 = performance.now();
      const cov = F.coverage(strata, S.budget, S.labels, ops, truth, 200, 101);
      const ms = performance.now() - t0;
      const row = (name, c, col) => `<tr><td>${name}</td><td style="width:45%"><div style="position:relative;height:8px;background:#f1f5f9;border-radius:4px"><div style="height:8px;width:${(c.coverage * 100).toFixed(1)}%;background:${col};border-radius:4px"></div><div style="position:absolute;left:95%;top:-4px;height:16px;border-left:2px dashed #2e2e2e"></div></div></td><td class="num"><b>${pct(c.coverage, 0)}</b></td><td class="num">${(c.width * 100).toFixed(1)} pts</td></tr>`;
      card.innerHTML = `<div class="tbl-tools"><div style="font-weight:600;font-size:15px">Interval Honesty Check</div><div class="search">${I('info')}200 resamples · ${Math.min(S.budget, idxOf().length)} replays · ${S.labels} reviews · ${ms.toFixed(0)} ms</div></div>
        <table class="tbl"><thead><tr><th>Method</th><th>Share of 95% intervals containing the truth (target 95%)</th><th class="num">Coverage</th><th class="num">Avg width</th></tr></thead><tbody>
        ${row('Judge only', cov.judge, '#9ca3af')}${row('Review only', cov.human, '#eab308')}${row('Judge + review (PPI++)', cov.ppi, '#388eff')}</tbody></table>`;
      icons();
    }, 30);
  }

  function renderExample(sample, ops) {
    const cand = CF.candidates.find((c) => c.id === S.cand);
    const pool = [];
    Object.entries(sample).forEach(([k, s]) => { s.labeled.forEach((i) => pool.push({ i, k, labeled: true })); s.unlabeled.forEach((i) => pool.push({ i, k, labeled: false })); });
    const rank = (p) => (p.k !== 'booked' && CF.outcomes[S.cand][p.i][0] === 1 ? 0 : p.k === 'booked' && CF.outcomes[S.cand][p.i][0] === 0 ? 1 : 2);
    pool.sort((a, b) => rank(a) - rank(b) || a.i - b.i);
    if (!pool.length) { $('example').innerHTML = '<div class="ch muted">No diverged replays in this sample.</div>'; return; }
    const p = pool[S.ex % pool.length];
    const c = calls[p.i];
    const rep = CF.replays[S.cand][c.id];
    const from = rep.from;
    const orig = c.turns.slice(Math.max(0, from - 1));
    const repl = (from > 0 ? [c.turns[from - 1]] : []).concat(rep.turns);
    const ln = (t, cls) => `<div class="ln ${t[0]} ${cls}"><b>${t[0] === 'agent' ? 'Avoca AI' : 'Customer'}</b>${esc(t[1])}</div>`;
    const f = ops.judge(p.i), y = ops.human(p.i);
    $('example').innerHTML = `
      <div class="tbl-tools"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><span style="font-weight:600;font-size:15px">Replay ${c.id}</span><span class="muted">${esc(custName[c.cust])} · ${esc(stratumLabel(p.k))}</span>
        <span class="pill ${c.booked ? 'green' : 'red'}">Original: ${c.booked ? 'Booked' : 'Unbooked'}</span><span class="pill blue">Judge: ${pct(f, 0)} likely to book</span>
        ${p.labeled ? `<span class="pill ${y ? 'green' : 'red'}">Human review: ${y ? 'would book' : 'would not book'}</span>` : '<span class="pill gray">Not reviewed</span>'}</div>
        <button class="btn" id="nextEx">Next replay ${I('arrow-right')}</button></div>
      <div class="diff" style="padding:14px">
        <div class="col"><h4>Original call</h4>${orig.map((t, k) => ln(t, k === (from > 0 ? 1 : 0) ? 'old' : '')).join('')}</div>
        <div class="col"><h4>Replay · ${esc(cand.short)}</h4>${repl.map((t, k) => ln(t, k === (from > 0 ? 1 : 0) ? 'new' : '')).join('')}</div>
      </div>`;
    $('nextEx').onclick = () => { S.ex++; renderExample(sample, ops); icons(); };
  }

  const rerender = () => { S.ex = 0; renderReplay(idxOf(), lostResults(idxOf())); };
  $('budget').oninput = (e) => { S.budget = +e.target.value; rerender(); };
  $('labels').oninput = (e) => { S.labels = +e.target.value; rerender(); };
  $('resample').onclick = () => { S.seed = (S.seed * 48271) % 2147483647; rerender(); };

  renderAll();
  icons();
})();
