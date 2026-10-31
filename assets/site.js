/* PSSA — shared behaviors: header state, scroll progress, reveal-on-scroll. */
(function () {
  var doc = document, root = doc.documentElement;
  var header = doc.querySelector('[data-header]');
  var progress = doc.querySelector('[data-scroll-progress]');

  function onScroll() {
    var y = window.scrollY || root.scrollTop || 0;
    if (header) header.classList.toggle('is-scrolled', y > 8);
    if (progress) {
      var height = root.scrollHeight - window.innerHeight;
      progress.style.transform = 'scaleX(' + (height > 0 ? Math.min(y / height, 1) : 0) + ')';
    }
  }
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  /* reveal-on-scroll */
  [].forEach.call(doc.querySelectorAll('[data-stagger]'), function (g) {
    [].forEach.call(g.querySelectorAll('.reveal'), function (k, i) {
      k.style.setProperty('--d', Math.min(i * 80, 480) + 'ms');
    });
  });
  root.classList.add('reveal-ready');
  var revealEls = [].slice.call(doc.querySelectorAll('.reveal'));
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          e.target.classList.add('is-in');
          io.unobserve(e.target);
        }
      });
    }, { threshold: .12, rootMargin: '0px 0px -8% 0px' });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add('is-in'); });
  }

  /* number counters */
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function fmt(v, el) {
    var d = +(el.getAttribute('data-decimals') || 0), s = el.getAttribute('data-suffix') || '';
    return v.toFixed(d) + s;
  }
  function runCount(el) {
    var t = parseFloat(el.getAttribute('data-count')); if (isNaN(t)) return;
    if (reduced) { el.textContent = fmt(t, el); return; }
    var t0 = null;
    function tick(ts) {
      if (!t0) t0 = ts;
      var p = Math.min((ts - t0) / 1400, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(t * e, el);
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }
  if ('IntersectionObserver' in window) {
    var cio = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { runCount(e.target); cio.unobserve(e.target); }
      });
    }, { threshold: .5 });
    [].forEach.call(doc.querySelectorAll('[data-count]'), function (el) { cio.observe(el); });
  } else {
    [].forEach.call(doc.querySelectorAll('[data-count]'), runCount);
  }
})();
