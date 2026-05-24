# Introduction

JAXVacua provides a unified, JAX-native pipeline from Calabi–Yau
compactification data to four-dimensional flux-vacuum solutions.
The introduction chapters below cover the physics and mathematics
that the rest of the documentation builds on. This page sketches the
end-to-end user workflow at a glance.

## The user workflow at a glance

The diagram below traces the typical use of the package from
geometric input to vacuum analysis. The four numbered stages map
onto the section captions of [this overview tutorial](../notebooks/01_basics/02_jaxvacua_overview).

```{raw} html
<div class="jx-fig mathjax_process" id="f1-workflow">
  <div class="jx-chart" id="f1-chart">

    <svg class="jx-asvg" id="f1-asvg">
      <defs>
        <marker id="f1-ab" markerWidth="8" markerHeight="6" refX="7.5" refY="3" orient="auto">
          <polygon points="0 0,8 3,0 6" fill="#2B5F8E"/></marker>
        <marker id="f1-ag" markerWidth="8" markerHeight="6" refX="7.5" refY="3" orient="auto">
          <polygon points="0 0,8 3,0 6" fill="#3A7D5A"/></marker>
        <marker id="f1-agr" markerWidth="8" markerHeight="6" refX="7.5" refY="3" orient="auto">
          <polygon points="0 0,8 3,0 6" fill="#999"/></marker>
      </defs>
    </svg>

    <div class="jx-title">jaxvacua &mdash; Workflow Overview</div>

    <!-- ══ STEP 1 ═════════════════════════════════════════════ -->
    <div class="jx-step">
      <div class="jx-step-hdr blue">
        <span class="lbl">Step 1 &nbsp;—&nbsp; Geometry Input</span>
        <span class="rl"></span>
      </div>

      <div class="jx-s1-row">
        <div class="bx lb" id="f1-src-ct">
          <div class="t">CYTools</div>
          <div class="d">Polytope / Triangulation<br>CalabiYau object</div>
        </div>
        <div class="bx lb" id="f1-src-dc">
          <div class="t">Python dictionary</div>
          <div class="d">$\kappa_{ijk}$,&nbsp; $a$-matrix,&nbsp; $c_2$,&nbsp; $\chi$<br>GV / GW invariants</div>
        </div>
        <div class="bx lb" id="f1-src-fi">
          <div class="t">Saved model file</div>
          <div class="d">Pickle / zipped pickle<br>from prior computation</div>
        </div>
        <div class="bx lb" id="f1-src-hf">
          <div class="t">Hugging Face</div>
          <div class="d">stringforge cy-database<br><code>aschachner/cy-database</code></div>
        </div>
      </div>

      <div class="gap-md"></div>

      <div class="jx-s1-bottom">
        <div class="bx db" id="f1-box-lcs">
          <div class="t">lcs_tree &nbsp;<span style="font-weight:400; font-size:10px; color:#7FBDE0; font-style:italic;">(preferred)</span></div>
          <div class="d">
            Central geometry data container &nbsp;&middot;&nbsp; JAX pytree<br>
            <code>from_cytools()&nbsp;&nbsp;&middot;&nbsp;&nbsp;from_dict()&nbsp;&nbsp;&middot;&nbsp;&nbsp;from_file()</code>
          </div>
        </div>

        <div class="jx-or-div">
          <span class="or-l"></span>
          <span class="or-t">or</span>
          <span class="or-l"></span>
        </div>

        <div class="bx lb-alt" id="f1-src-dp">
          <div class="t">Direct input &nbsp;<span style="font-weight:400; font-size:10px; color:#6a96c0; font-style:italic;">(alternative)</span></div>
          <div class="d">
            Provide $\Pi(z)$ or $\mathcal{F}(X)$<br>
            directly as callable functions<br>
            <code>period_input=...</code><br>
            <code>prepotential_input=...</code>
          </div>
        </div>
      </div>
    </div>

    <div class="gap-md"></div>

    <!-- ══ STEP 2 ═════════════════════════════════════════════ -->
    <div class="jx-step">
      <div class="jx-step-hdr blue">
        <span class="lbl">Step 2 &nbsp;—&nbsp; Model Construction</span>
        <span class="rl"></span>
      </div>

      <div class="jx-s2-chain">

        <div class="bx lb2" id="f1-box-per" style="position:relative;">
          <div class="t">periods</div>
          <div class="d">
            Period vector $\Pi(z) = (\mathcal{F}_I,\, X^I)$
            &nbsp;&middot;&nbsp;
            Prepotential $\mathcal{F}(X)$<br>
            Kähler potential
            $K = -\!\log\!\bigl(-i\,\Pi^\dagger \Sigma\,\Pi\bigr)$
            &nbsp;&middot;&nbsp;
            Gauge kinetic matrix $\mathcal{N}_{IJ}$
          </div>
          <span class="inh">inherits ↓</span>
        </div>

        <div class="gap-sm"></div>

        <div class="bx lb2" id="f1-box-css" style="position:relative;">
          <div class="t">css</div>
          <div class="d">
            Kähler metric
            $G_{i\bar\jmath} = \partial_i \partial_{\bar\jmath} K$
            &nbsp;&middot;&nbsp;
            ISD matrix $M$<br>
            Complex-structure sector
            &nbsp;&middot;&nbsp;
            Special-Kähler structure
          </div>
          <span class="inh">inherits ↓</span>
        </div>

        <div class="gap-sm"></div>

        <div class="bx lb2" id="f1-box-feft" style="position:relative;">
          <div class="t">FluxEFT</div>
          <div class="d">
            Superpotential $W = \int G_3 \wedge \Omega = (F - \tau H) \cdot \Pi$
            &nbsp;&middot;&nbsp;
            F-terms $D_I W = \partial_I W + (\partial_I K)\,W$<br>
            Scalar potential $V$
            &nbsp;&middot;&nbsp;
            Tadpole $N_{\rm flux} = f^T \Sigma\, h$
            &nbsp;&middot;&nbsp;
            D3-charge constraint $N_{\rm flux} \leq Q_{\rm O3}$
          </div>
          <span class="inh">inherits ↓</span>
        </div>

        <div class="gap-sm"></div>

        <div class="bx db" id="f1-box-fvf">
          <div class="t">FluxVacuaFinder</div>
          <div class="d">
            Kähler cone &nbsp;&middot;&nbsp; Conifold loci &nbsp;&middot;&nbsp; EFT validity constraints<br>
            Moduli-space limits &nbsp;&middot;&nbsp; Perturbatively flat vacua
            &nbsp;&middot;&nbsp;
            Vacuum-search entry point
          </div>
        </div>

      </div>
    </div>

    <div class="gap-md"></div>

    <!-- ══ STEP 3 ═════════════════════════════════════════════ -->
    <div class="jx-step">
      <div class="jx-step-hdr green">
        <span class="lbl">Step 3 &nbsp;—&nbsp; Vacuum Search</span>
        <span class="rl"></span>
        <div class="bx gh jx-fig-freezer" id="f1-box-frz">
          <div class="t">Freezer</div>
          <div class="d">
            Freeze heavy moduli<br>
            Reduced EFT for<br>light-sector fields
          </div>
        </div>
      </div>

      <div class="jx-s3-samp">
        <div class="bx lg" id="f1-box-samp">
          <div class="t">data_sampler</div>
          <div class="d">
            Kähler-cone moduli sampling
            &nbsp;&middot;&nbsp;
            Axion / dilaton bounds<br>
            Tadpole constraint $N_{\rm flux} \leq N_{\max}$
            &nbsp;&middot;&nbsp;
            Vmapped scan kernels
          </div>
        </div>
      </div>

      <div class="gap-md"></div>

      <div class="jx-s3-split">
        <div class="bx lg" id="f1-box-sto">
          <div class="t">Stochastic ISD sampling</div>
          <div class="d">
            ISD completion: $f = s\,M\,\sigma\, h + c_0\, h$<br>
            Random $h \in \mathbb{Z}^{2 (h^{1,2}+1)}$
            with $N_{\rm flux} \leq N_{\max}$<br>
            <code>data_sampler.ISD_sampling()</code>
          </div>
        </div>
        <div class="bx lg" id="f1-box-sys">
          <div class="t">Systematic flux enumeration</div>
          <div class="d">
            Eigenvalue bounding boxes for $h$<br>
            Integer lattice scan + ISD filter<br>
            <code>bounded_fluxes.sample_bounded_fluxes()</code>
          </div>
        </div>
      </div>
    </div>

    <div class="gap-md"></div>

    <!-- ══ STEP 4 ═════════════════════════════════════════════ -->
    <div class="jx-step">
      <div class="jx-step-hdr green">
        <span class="lbl">Step 4 &nbsp;—&nbsp; Refinement &amp; Analysis</span>
        <span class="rl"></span>
      </div>

      <div class="jx-s4-wrap">
        <div class="bx dg" id="f1-box-ref">
          <div class="t">Vacuum refinement</div>
          <div class="d">
            Solve $D_I W = 0 \;\Longrightarrow\; (z^*,\, \tau^*)$<br>
            Newton method
            &nbsp;&middot;&nbsp;
            <code>FluxVacuaFinder.newton_method_flux_vacua()</code>
            &nbsp;&middot;&nbsp;
            <code>scipy.optimize.root</code><br>
            Hessian / mass matrix via <code>FluxEFT.hessian</code>
            and <code>FluxEFT.mass_matrix</code>
          </div>
        </div>
      </div>

      <div class="gap-sm"></div>

      <div class="jx-out-bar">
        <strong>Output:</strong>&nbsp;
        Flux vacua $(z^*,\, \tau^*,\, f,\, h)$,&nbsp;
        residual,&nbsp; $|W_0|$,&nbsp; $g_s$,&nbsp; tadpole,&nbsp; mass spectrum,&nbsp; EFT validity checks
        &nbsp;&middot;&nbsp;
        optionally written to the stringforge vacua storage layer.
      </div>
    </div>

  </div><!-- .jx-chart -->
</div><!-- .jx-fig -->

<script>
(function() {
  function drawF1() {
    var chart = document.getElementById('f1-chart');
    var svg   = document.getElementById('f1-asvg');
    if (!chart || !svg) return;

    // clear any previously-drawn arrows (in case of resize / re-layout)
    while (svg.lastChild) {
      if (svg.lastChild.tagName === 'defs') break;
      svg.removeChild(svg.lastChild);
    }

    var cr = chart.getBoundingClientRect();
    var H  = chart.scrollHeight;
    svg.setAttribute('height', H);
    svg.setAttribute('viewBox', '0 0 ' + cr.width + ' ' + H);

    function g(id) {
      var el = document.getElementById(id);
      if (!el) return null;
      var r = el.getBoundingClientRect();
      return {
        cx: r.left - cr.left + r.width  / 2,
        cy: r.top  - cr.top  + r.height / 2,
        t:  r.top    - cr.top,
        b:  r.bottom - cr.top,
        l:  r.left   - cr.left,
        r:  r.right  - cr.left,
        w:  r.width,
        h:  r.height
      };
    }
    function seg(x1, y1, x2, y2, col, dash) {
      var el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      el.setAttribute('x1', x1); el.setAttribute('y1', y1);
      el.setAttribute('x2', x2); el.setAttribute('y2', y2);
      el.setAttribute('stroke', col);
      el.setAttribute('stroke-width', '1.6');
      if (dash) el.setAttribute('stroke-dasharray', dash);
      svg.appendChild(el);
    }
    function arr(x1, y1, x2, y2, mid, col, dash) {
      var el = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      el.setAttribute('x1', x1); el.setAttribute('y1', y1);
      el.setAttribute('x2', x2); el.setAttribute('y2', y2);
      el.setAttribute('stroke', col);
      el.setAttribute('stroke-width', '1.6');
      el.setAttribute('marker-end', 'url(#' + mid + ')');
      if (dash) el.setAttribute('stroke-dasharray', dash);
      svg.appendChild(el);
    }
    function txt(x, y, s, col, style) {
      var el = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      el.setAttribute('x', x); el.setAttribute('y', y);
      el.setAttribute('font-size', '9');
      el.setAttribute('fill', col || '#999');
      if (style) el.setAttribute('font-style', style);
      el.textContent = s;
      svg.appendChild(el);
    }

    var B = '#2B5F8E', G = '#3A7D5A', GR = '#999';

    // 1. source boxes → bus → lcs_tree
    var srcs = ['f1-src-ct','f1-src-dc','f1-src-fi','f1-src-hf'].map(g);
    var lcs  = g('f1-box-lcs');
    var per  = g('f1-box-per');
    if (lcs) {
      var busY = lcs.t - 15;
      srcs.forEach(function(s) { if (s) seg(s.cx, s.b, s.cx, busY, B); });
      if (srcs[0] && srcs[srcs.length - 1]) seg(srcs[0].cx, busY, srcs[srcs.length - 1].cx, busY, B);
      arr(lcs.cx, busY, lcs.cx, lcs.t - 2, 'f1-ab', B);

      // 2. lcs_tree → periods (preferred path, solid)
      if (per) {
        arr(lcs.cx, lcs.b, per.cx, per.t - 2, 'f1-ab', B);
        txt(lcs.cx + 6, lcs.b + (per.t - lcs.b) * 0.48, 'preferred', '#5A9EC8', 'italic');
      }
    }

    // 2b. Direct input → periods (alternative, dashed diagonal)
    var dp = g('f1-src-dp');
    if (dp && per) {
      arr(dp.cx, dp.b, per.cx + 8, per.t - 2, 'f1-ab', B, '5,3');
      var _dx = (per.cx + 8) - dp.cx, _dy = (per.t - 2) - dp.b;
      var _len = Math.sqrt(_dx * _dx + _dy * _dy) || 1;
      txt((dp.cx + per.cx + 8) / 2 + 9 * _dy / _len,
          (dp.b + per.t - 2) / 2 - 9 * _dx / _len,
          'alternative', '#6a96c0', 'italic');
    }

    // 3. inheritance chain (4 layers)
    [['f1-box-per','f1-box-css'],
     ['f1-box-css','f1-box-feft'],
     ['f1-box-feft','f1-box-fvf']].forEach(function(p) {
      var ga = g(p[0]), gb = g(p[1]);
      if (ga && gb) arr(ga.cx, ga.b, gb.cx, gb.t - 2, 'f1-ab', B);
    });

    // 4. FluxVacuaFinder → data_sampler
    var fvf  = g('f1-box-fvf');
    var samp = g('f1-box-samp');
    if (fvf && samp) arr(fvf.cx, fvf.b, samp.cx, samp.t - 2, 'f1-ag', G);

    // 5. sampler → split → stochastic + systematic
    var sto = g('f1-box-sto');
    var sys = g('f1-box-sys');
    if (samp && sto && sys) {
      var spY = samp.b + 15;
      seg(samp.cx, samp.b, samp.cx, spY, G);
      seg(sto.cx, spY, sys.cx, spY, G);
      arr(sto.cx, spY, sto.cx, sto.t - 2, 'f1-ag', G);
      arr(sys.cx, spY, sys.cx, sys.t - 2, 'f1-ag', G);

      // 6. stochastic + systematic → refinement
      //    Merge BEFORE the Step 4 step-header (i.e., right below the
      //    Layer 3 boxes). A single arrow then carries the merged flow
      //    THROUGH the Step 4 header and into the refinement box.
      var ref = g('f1-box-ref');
      if (ref) {
        var merY = Math.max(sto.b, sys.b) + 16;
        seg(sto.cx, sto.b, sto.cx, merY, G);
        seg(sys.cx, sys.b, sys.cx, merY, G);
        seg(sto.cx, merY, sys.cx, merY, G);
        arr(ref.cx, merY, ref.cx, ref.t - 2, 'f1-ag', G);
      }
    }

    // 7. FluxVacuaFinder → Freezer → data_sampler (dashed, optional)
    var frz = g('f1-box-frz');
    if (frz && fvf && samp) {
      // FluxVacuaFinder right edge → right to frz.cx → down to frz top
      var pl1 = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      pl1.setAttribute('points',
        fvf.r + ',' + fvf.cy + ' ' + frz.cx + ',' + fvf.cy + ' ' + frz.cx + ',' + (frz.t - 2));
      pl1.setAttribute('stroke', GR);
      pl1.setAttribute('stroke-width', '1.5');
      pl1.setAttribute('stroke-dasharray', '5,3');
      pl1.setAttribute('fill', 'none');
      pl1.setAttribute('marker-end', 'url(#f1-agr)');
      svg.appendChild(pl1);

      // freezer bottom → down to samp.cy level → left into samp right side
      var pl2 = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      pl2.setAttribute('points',
        frz.cx + ',' + frz.b + ' ' + frz.cx + ',' + samp.cy + ' ' + (samp.r + 12) + ',' + samp.cy);
      pl2.setAttribute('stroke', GR);
      pl2.setAttribute('stroke-width', '1.5');
      pl2.setAttribute('stroke-dasharray', '5,3');
      pl2.setAttribute('fill', 'none');
      pl2.setAttribute('marker-end', 'url(#f1-agr)');
      svg.appendChild(pl2);

      txt(fvf.r + 8, fvf.cy - 6, 'optional', GR, 'italic');
    }
  }

  function setupF1() {
    if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {
      window.MathJax.startup.promise.then(function() {
        requestAnimationFrame(function() { requestAnimationFrame(drawF1); });
      });
    } else {
      requestAnimationFrame(drawF1);
    }
  }

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setupF1();
  } else {
    window.addEventListener('load', setupF1);
  }
  // Re-draw on viewport resize
  var _t;
  window.addEventListener('resize', function() {
    clearTimeout(_t);
    _t = setTimeout(function() {
      requestAnimationFrame(function() { requestAnimationFrame(drawF1); });
    }, 100);
  });
})();
</script>
```

The four stages are:

1. **Geometry input** — load topological data through any of four
   on-ramps (CYTools polytope, the
   [stringforge cy-database](https://huggingface.co/datasets/aschachner/cy-database),
   a local CICY identifier, or an explicit dictionary). All four
   feed an `lcs_tree` — JAXVacua's data interface for a
   Calabi–Yau threefold.
2. **Build the EFT** — the linear pipeline `periods → css →
   FluxEFT → FluxVacuaFinder` constructs the period vector,
   complex-structure Kähler geometry, GVW superpotential, and
   vacuum solver from the `lcs_tree`.
3. **Search for vacua** — `data_sampler` provides ISD-biased
   initial guesses; `bounded_fluxes` enumerates fluxes inside a
   box subject to physical constraints. Both feed
   `FluxVacuaFinder`'s Newton solver.
4. **Analyse and store** — extract observables ($\lvert W_0
   \rvert$, $g_s$, tadpole, mass spectrum) and persist them to a
   local or community vacua vault via the `stringforge.vacua_writer`
   layer. The optional `Freezer` branch produces a reduced EFT
   when heavy moduli are integrated out.

## Reading order

The chapters that follow in the **Introduction** caption of the master
TOC cover the physics in the order you will encounter it when
constructing a model: supergravity background, Calabi–Yau geometries,
flux compactifications, moduli stabilisation, periods, and
perturbatively flat vacua. For the corresponding code-level
walkthroughs, see the **Tutorials — Basics** chapter.
