/* PSSA — technical page: scroll-in for sections, chart draw, result bars; loss tooltips. */
(function () {
  var doc = document;
  var targets = [].slice.call(doc.querySelectorAll('.section, .chart-wrap, .results-table tr'));
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { threshold: .25 });
    targets.forEach(function (el) { io.observe(el); });
  } else {
    targets.forEach(function (el) { el.classList.add('in'); });
  }

  var tip = doc.querySelector('[data-loss-tip]');
  var wrap = doc.querySelector('.chart-wrap');
  if (tip && wrap) {
    [].forEach.call(doc.querySelectorAll('.loss-dot'), function (d) {
      d.addEventListener('pointerenter', function () {
        tip.textContent = 'checkpoint ' + d.dataset.ck + ' \u00b7 loss ' + d.dataset.loss;
        var r = d.getBoundingClientRect(), w = wrap.getBoundingClientRect();
        tip.style.left = (r.left - w.left + r.width / 2) + 'px';
        tip.style.top = (r.top - w.top) + 'px';
        tip.style.opacity = 1;
      });
      d.addEventListener('pointerleave', function () { tip.style.opacity = 0; });
    });
  }
})();
