const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <nav class="fixed w-full z-50 glass px-6 py-4 flex justify-between items-center">
    <div class="flex items-center gap-2">
      <div class="w-8 h-8 bg-gradient-to-br from-blue-500 to-emerald-500 rounded-lg flex items-center justify-center font-bold text-white">C</div>
      <span class="text-xl font-bold tracking-tight text-white">ChapatiLM</span>
    </div>
    <div class="hidden md:flex gap-8 text-sm font-medium text-slate-400">
      <a href="#" class="hover:text-white transition-colors">Architecture</a>
      <a href="#" class="hover:text-white transition-colors">Benchmarks</a>
      <a href="#" class="hover:text-white transition-colors">Docs</a>
    </div>
    <button class="bg-blue-600 hover:bg-blue-700 px-5 py-2 rounded-full text-sm font-semibold transition-all shadow-lg shadow-blue-500/20 text-white">
      Get Started
    </button>
  </nav>

  <main>
    <section class="pt-32 pb-20 px-6 max-w-7xl mx-auto text-center">
      <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-400 text-xs font-medium mb-8">
        <span class="relative flex h-2 w-2">
          <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
          <span class="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
        </span>
        ChapatiLM v1.0.0 is now live
      </div>
      <h1 class="text-5xl md:text-7xl font-extrabold mb-6 tracking-tight text-white">
        Next-Gen ML Core <br/>
        <span class="gradient-text">Efficient by Design.</span>
      </h1>
      <p class="text-slate-400 text-lg md:text-xl max-w-2xl mx-auto mb-10 leading-relaxed">
        ChapatiLM provides the premium frontend interface for our neural engine,
        delivering real-time orchestration and high-performance visualizations
        for your machine learning workflows.
      </p>
      <div class="flex flex-col sm:flex-row gap-4 justify-center">
        <button class="bg-white text-black px-8 py-4 rounded-xl font-bold hover:bg-slate-200 transition-all">
          Deploy Architecture
        </button>
        <button class="glass px-8 py-4 rounded-xl font-bold hover:bg-white/5 transition-all text-white">
          View Benchmarks
        </button>
      </div>
    </section>

    <section class="py-20 px-6 max-w-7xl mx-auto">
      <div class="grid md:grid-cols-3 gap-8">
        <div class="p-8 rounded-3xl glass hover:border-blue-500/30 transition-all group">
          <div class="w-12 h-12 bg-blue-500/10 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-blue-500/20 transition-colors">
            <svg class="w-6 h-6 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
          </div>
          <h3 class="text-xl font-bold mb-3 text-white">Neural Warp</h3>
          <p class="text-slate-400 leading-relaxed">Optimized SIMD-aligned memory management for lightning-fast model execution on standard CPUs.</p>
        </div>
        <div class="p-8 rounded-3xl glass hover:border-emerald-500/30 transition-all group">
          <div class="w-12 h-12 bg-emerald-500/10 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-emerald-500/20 transition-colors">
            <svg class="w-6 h-6 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
          </div>
          <h3 class="text-xl font-bold mb-3 text-white">Secure Sandbox</h3>
          <p class="text-slate-400 leading-relaxed">Isolated compute environments ensuring data integrity and stability during complex training runs.</p>
        </div>
        <div class="p-8 rounded-3xl glass hover:border-purple-500/30 transition-all group">
          <div class="w-12 h-12 bg-purple-500/10 rounded-2xl flex items-center justify-center mb-6 group-hover:bg-purple-500/20 transition-colors">
            <svg class="w-6 h-6 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
          </div>
          <h3 class="text-xl font-bold mb-3 text-white">XTRAIN Integrated</h3>
          <p class="text-slate-400 leading-relaxed">Seamless integration with XTRAIN backend for state-of-the-art mathematical reasoning models.</p>
        </div>
      </div>
    </section>

    <footer class="py-12 border-t border-white/5 px-6">
      <div class="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6">
        <div class="flex items-center gap-2">
          <div class="w-6 h-6 bg-slate-800 rounded flex items-center justify-center text-xs font-bold text-white">C</div>
          <span class="font-semibold text-slate-300">ChapatiLM Architecture</span>
        </div>
        <p class="text-slate-500 text-sm">© 2024 ChapatiLM Architecture. All rights reserved.</p>
        <div class="flex gap-6 text-slate-400 text-sm">
          <a href="#" class="hover:text-white transition-colors">Twitter</a>
          <a href="#" class="hover:text-white transition-colors">GitHub</a>
        </div>
      </div>
    </footer>
  </main>
`;