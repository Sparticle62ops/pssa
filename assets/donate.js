/* PSSA — donate page: SOL address copy button. */
(function () {
  var b = document.getElementById('copy-sol');
  var a = document.getElementById('sol-addr');
  if (!b || !a) return;
  b.addEventListener('click', function () {
    var addr = a.textContent.trim();
    function done() {
      var t = b.textContent;
      b.textContent = 'Copied';
      setTimeout(function () { b.textContent = t; }, 1600);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(addr).then(done, done);
    } else {
      var i = document.createElement('textarea');
      i.value = addr;
      document.body.appendChild(i);
      i.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(i);
      done();
    }
  });
})();
