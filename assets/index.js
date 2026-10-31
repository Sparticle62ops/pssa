/* PSSA — index page behaviors: phase walkthrough, loss chart, token schematic, bench bars.
   Shared behaviors (header, reveal, counters, stagger) live in site.js. */
(function () {
  var doc = document, root = doc.documentElement;
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var hasIO = 'IntersectionObserver' in window;

  /* ---------------- walkthrough ---------------- */
  var panels = [].slice.call(doc.querySelectorAll('.phase-panel'));
  if (panels.length) {
    root.classList.add('walk-ready');
    var tabs = [].slice.call(doc.querySelectorAll('.step-tab'));
    var nodes = [].slice.call(doc.querySelectorAll('[data-node]'));
    var callout = doc.querySelector('[data-callout]');
    var walkStatus = doc.querySelector('[data-walk-status]');
    var countPhase = doc.querySelector('[data-count-phase]');
    var prevBtn = doc.querySelector('[data-prev]');
    var nextBtn = doc.querySelector('[data-next]');
    var narrateBtn = doc.querySelector('[data-narration]');
    var cur = 0, narrateOn = false;
    function pad(n){ return (n < 10 ? '0' : '') + n; }
    var narr = window.speechSynthesis ? window.speechSynthesis : null;
    function phaseText(){
      var p = panels[cur]; if (!p) return '';
      var a = p.querySelector('h3'), b = p.querySelector('.phase-copy');
      return ((a ? a.textContent : '') + '. ' + (b ? b.textContent : '')).trim();
    }
    function speakPhase(){
      if (!narr) return;
      narr.cancel();
      if (!narrateOn) return;
      var u = new SpeechSynthesisUtterance(phaseText());
      u.rate = 1.02; u.pitch = 1;
      narr.speak(u);
    }
    function setPhase(i){
      i = Math.max(0, Math.min(panels.length - 1, i)); cur = i;
      panels.forEach(function (p, k) {
        if (k === i) p.removeAttribute('hidden'); else p.setAttribute('hidden', '');
        p.classList.toggle('is-active', k === i);
      });
      tabs.forEach(function (t, k) {
        t.setAttribute('aria-pressed', k === i ? 'true' : 'false');
        t.classList.toggle('is-active', k === i);
      });
      nodes.forEach(function (n) {
        var ph = (n.getAttribute('data-phases') || '').trim().split(/\s+/);
        var on = ph.indexOf(String(i)) !== -1;
        n.classList.toggle('is-active', on);
        n.setAttribute('aria-expanded', on ? 'true' : 'false');
      });
      var lbl = (tabs[i] ? tabs[i].textContent : '').replace(/^\s*\d+\s*/, '').trim().toLowerCase();
      if (callout) callout.textContent = pad(i + 1) + ' \u00b7 ' + lbl;
      if (walkStatus) walkStatus.textContent = 'Phase ' + (i + 1) + ' of ' + panels.length + ': ' + lbl;
      if (countPhase) countPhase.textContent = pad(i + 1) + ' / ' + pad(panels.length);
      if (prevBtn) prevBtn.disabled = (i === 0);
      if (nextBtn) nextBtn.disabled = (i === panels.length - 1);
      speakPhase();
    }
    tabs.forEach(function (t) { t.addEventListener('click', function () { setPhase(+t.getAttribute('data-step')); }); });
    nodes.forEach(function (n) { n.addEventListener('click', function () { setPhase(+n.getAttribute('data-target')); }); });
    if (prevBtn) prevBtn.addEventListener('click', function () { setPhase(cur - 1); });
    if (nextBtn) nextBtn.addEventListener('click', function () { setPhase(cur + 1); });
    if (narrateBtn) narrateBtn.addEventListener('click', function () {
      narrateOn = !narrateOn;
      narrateBtn.setAttribute('aria-pressed', narrateOn ? 'true' : 'false');
      speakPhase();
    });
    setPhase(0);
  }

  /* ---------------- loss chart ---------------- */
  var LOSS = [5.732707,5.085828,4.878240,4.732464,4.617316,4.590629,4.556216,4.446911,4.388450,4.446946,4.368871,4.384911,4.278566,4.321767,4.292169,4.186657,4.207503,4.197082,4.153711,4.185269,4.167721,4.125303,4.122400,4.087714,4.221450,4.105929,4.088992,4.101552,4.076447,4.069941,4.071954,3.988346,4.028232,4.020699,4.039323,3.957713,3.959645,4.006465,4.002451,3.972586,3.970350,3.934968,3.944302,4.003730,3.953108,3.938739,3.883540,3.937216,3.906847,3.961127,3.914990,3.846320,3.896660,3.939427,3.841899,3.886533,3.936252,3.919427,3.918769,3.972387,3.928461,3.892412,3.931151,3.982083];
  var frame = doc.querySelector('[data-chart-frame]');
  if (frame && LOSS.length === 64) {
    var dyn = frame.querySelector('.loss-line.dynamic');
    var area = frame.querySelector('.loss-area');
    var cursor = frame.querySelector('[data-chart-cursor]');
    var dot = frame.querySelector('[data-chart-dot]');
    var rLoss = doc.querySelector('[data-readout-loss]');
    var rMeta = doc.querySelector('[data-readout-meta]');
    var rStep = doc.querySelector('[data-readout-step]');
    var curIdx = 0, drawn = false;
    var d = LOSS.map(function (v, i) {
      var x = (i / (LOSS.length - 1)) * 1000;
      var y = Math.max(0, Math.min(420, 20 + (5.75 - v) / 0.5 * 100));
      return (i ? 'L' : 'M') + x.toFixed(1) + ' ' + y.toFixed(1);
    }).join(' ');
    if (dyn) dyn.setAttribute('d', d);
    if (area) area.setAttribute('d', d + ' L1000 420 L0 420 Z');
    function showAt(i){
      curIdx = Math.max(0, Math.min(LOSS.length - 1, i));
      var x = (curIdx / (LOSS.length - 1)) * 100;
      var y = Math.max(0, Math.min(420, 20 + (5.75 - LOSS[curIdx]) / 0.5 * 100)) / 420 * 100;
      if (cursor){ cursor.style.left = x + '%'; cursor.classList.add('is-visible'); }
      if (dot){ dot.style.left = x + '%'; dot.style.top = y + '%'; dot.classList.add('is-visible'); }
      if (rLoss) rLoss.textContent = LOSS[curIdx].toFixed(3);
      if (rMeta) rMeta.textContent = 'checkpoint ' + (curIdx + 1) + ' of 64';
      if (rStep) rStep.textContent = ((curIdx + 1) * 0.199).toFixed(1) + 'M tokens seen';
    }
    function hide(){ if (cursor) cursor.classList.remove('is-visible'); if (dot) dot.classList.remove('is-visible'); }
    frame.addEventListener('pointermove', function (ev) {
      var r = frame.getBoundingClientRect();
      showAt(Math.round((ev.clientX - r.left) / r.width * (LOSS.length - 1)));
    });
    frame.addEventListener('pointerleave', hide);
    frame.addEventListener('keydown', function (ev) {
      if (ev.key === 'ArrowRight'){ showAt(curIdx + 1); ev.preventDefault(); }
      else if (ev.key === 'ArrowLeft'){ showAt(curIdx - 1); ev.preventDefault(); }
    });
    frame.addEventListener('blur', hide);
    function draw(){
      if (drawn) return; drawn = true;
      frame.classList.add('chart-ready');
      if (reduced){ frame.classList.add('is-drawn'); return; }
      requestAnimationFrame(function () { requestAnimationFrame(function () { frame.classList.add('is-drawn'); }); });
    }
    if (hasIO) {
      var gio = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting){ draw(); gio.unobserve(e.target); } });
      }, { threshold: .3 });
      gio.observe(frame);
    } else draw();
  }

  /* ---------------- token schematic ---------------- */
  var demo = doc.querySelector('[data-demo]');
  if (demo) {
    var form = demo.querySelector('[data-demo-form]');
    var input = demo.querySelector('[data-demo-input]');
    var runBtn = demo.querySelector('[data-demo-run]');
    var statusEl = demo.querySelector('[data-demo-status]');
    var chipsEl = demo.querySelector('[data-chips]');
    var bank = demo.querySelector('[data-bank]');
    var fired = demo.querySelector('[data-fired]');
    var firedEmpty = demo.querySelector('[data-fired-empty]');
    var gateVal = demo.querySelector('[data-gate-value]');
    var gateFill = demo.querySelector('[data-gate-fill]');
    var cells = [].slice.call(demo.querySelectorAll('.state-cell'));
    var stages = [].slice.call(demo.querySelectorAll('.stage'));
    function hstr(s){
      var hsh = 2166136261;
      for (var i = 0; i < s.length; i++){ hsh ^= s.charCodeAt(i); hsh = (hsh * 16777619) >>> 0; }
      return hsh;
    }
    function frac(n, salt){ return ((hstr(n + '\u00b7' + salt) % 10000) / 10000); }
    function hex(n){ var s = n.toString(16).toUpperCase(); while (s.length < 4) s = '0' + s; return '#' + s; }
    function tokenId(w){ return hstr(w.toLowerCase()) % 2048; }
    function tokenize(str){
      var m = str.match(/[A-Za-z0-9']+|[.,!?;:'"()-]/g) || [];
      return m.slice(0, 24);
    }
    function renderChips(toks){
      if (!chipsEl) return;
      chipsEl.innerHTML = '';
      toks.forEach(function (w) {
        var sp = doc.createElement('span'); sp.className = 'token-chip';
        var a = doc.createElement('span'); a.className = 'chip-text'; a.textContent = w;
        var b = doc.createElement('span'); b.className = 'chip-id'; b.textContent = hex(tokenId(w));
        sp.appendChild(a); sp.appendChild(b); chipsEl.appendChild(sp);
      });
    }
    var bankCells = [];
    if (bank && !bank.childNodes.length) {
      for (var s = 0; s < 512; s++) {
        var c = doc.createElement('span'); c.className = 'slot-cell';
        bank.appendChild(c); bankCells.push(c);
      }
    }
    var state = cells.map(function (cell) {
      var q = cell.querySelector('.state-num');
      var t = (q ? q.textContent : '+0') || '+0';
      return parseFloat(t.replace('\u2212', '-')) || 0;
    });
    function setState(cell, v){
      var fill = cell.querySelector('.state-fill'); if (!fill) return;
      var mag = Math.min(Math.abs(v), 1) * 50;
      if (v >= 0){ fill.classList.remove('neg'); fill.style.top = ''; fill.style.height = mag + '%'; fill.style.bottom = '50%'; }
      else { fill.classList.add('neg'); fill.style.bottom = ''; fill.style.height = mag + '%'; fill.style.top = '50%'; }
      var num = cell.querySelector('.state-num');
      if (num) num.textContent = (v >= 0 ? '+' : '\u2212') + Math.abs(v).toFixed(2);
    }
    function setGate(g){
      if (gateVal) gateVal.textContent = g.toFixed(2);
      if (gateFill) gateFill.style.width = (g * 100).toFixed(1) + '%';
    }
    function stageLive(k, text){
      var st = stages[k]; if (!st) return;
      var ph = st.querySelector('.placeholder'), chip = st.querySelector('.live-chip');
      if (ph) ph.textContent = text;
      if (chip) chip.textContent = text;
      st.classList.add('is-active');
    }
    function stageIdle(k){
      var st = stages[k]; if (!st) return;
      st.classList.remove('is-active');
    }
    function clearBank(){
      bankCells.forEach(function (c) { c.classList.remove('is-fired'); });
    }
    function fire(w){
      var wl = w.toLowerCase();
      var i1 = Math.floor(frac(wl, 'a') * 512), i2 = Math.floor(frac(wl, 'b') * 512);
      [i1, i2].forEach(function (si) {
        var c = bankCells[si]; if (c) c.classList.add('is-fired');
      });
      if (fired) {
        if (firedEmpty && firedEmpty.parentNode) firedEmpty.parentNode.removeChild(firedEmpty);
        var it = doc.createElement('span');
        it.textContent = 'slot ' + i1 + ' \u00b7 ' + w + '   slot ' + i2 + ' \u00b7 gate ' + frac(wl, 'g').toFixed(2);
        fired.appendChild(it);
      }
    }
    var running = false, toks = [];
    function step(k, total){
      stageLive(0, 'E[' + toks[k] + '] 256-d');
      stageLive(1, 'h update');
      stageLive(2, 'scan 512 keys');
      stageLive(3, 'logits + write');
      var wl = toks[k].toLowerCase();
      setGate(0.22 + 0.56 * frac(wl, 'g'));
      for (var j = 0; j < state.length; j++) {
        var delta = frac(wl + j, 'h') * 2 - 1;
        state[j] = Math.max(-1, Math.min(1, state[j] * 0.72 + delta * 0.55));
        setState(cells[j], state[j]);
      }
      fire(toks[k]);
      if (statusEl) statusEl.textContent = 'Running \u00b7 token ' + (k + 1) + ' of ' + total + ' \u2014 ' + toks[k];
      if (k + 1 < total && !reduced) {
        setTimeout(function () {
          stageIdle(0); stageIdle(1); stageIdle(2); stageIdle(3);
          step(k + 1, total);
        }, 460);
      } else {
        stageIdle(0); stageIdle(1); stageIdle(2); stageIdle(3);
        if (statusEl) statusEl.textContent = 'Done \u00b7 ' + total + ' tokens \u00b7 ' + (total * 2) + ' slot writes \u00b7 schematic only, no live inference.';
        running = false;
        if (runBtn) runBtn.removeAttribute('disabled');
      }
    }
    if (form) form.addEventListener('submit', function (ev) {
      ev.preventDefault();
      if (running) return;
      var txt = (input && input.value ? input.value : '').trim() || 'the river remembers its shape';
      toks = tokenize(txt);
      if (!toks.length){ if (statusEl) statusEl.textContent = 'Type a sentence first.'; return; }
      renderChips(toks);
      running = true;
      if (runBtn) runBtn.setAttribute('disabled', '');
      clearBank();
      if (fired && firedEmpty && firedEmpty.parentNode) firedEmpty.parentNode.removeChild(firedEmpty);
      step(0, toks.length);
    });
  }

  /* ---------------- bench bars ---------------- */
  var bench = doc.querySelector('[data-bench]');
  if (bench && hasIO && !reduced) {
    [].forEach.call(bench.querySelectorAll('.bar-fill'), function (f, i) {
      f.style.setProperty('--bar-delay', (i * 80) + 'ms');
    });
    bench.classList.add('motion-ready');
    var bio = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { bench.classList.add('is-inview'); bio.disconnect(); }
      });
    }, { threshold: .3 });
    bio.observe(bench);
  }
})();
