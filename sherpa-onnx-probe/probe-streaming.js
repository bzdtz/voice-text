/**
 * Probe: validate sherpa-onnx streaming zipformer for MeetingCopilot.
 *
 * Simulates the exact feed pattern the app uses today (AudioWorklet ->
 * 100ms / 1600-sample frames) and measures:
 *   - model load time
 *   - first-char latency (audio time at which the first partial appears)
 *   - real-time factor (RTF)
 *   - peak RSS memory
 *   - final transcript
 *
 * Usage:
 *   node probe-streaming.js                     # run all built-in audio files
 *   node probe-streaming.js --chunk 50          # 50ms feed frames
 *   node probe-streaming.js --wav <path> [--chunk 100]
 */
const path = require('path');
const sherpa = require('sherpa-onnx-node');

const MODELS = {
  zh14m: {
    label: 'streaming-zipformer-zh-14M (int8)',
    dir: path.join(__dirname, 'models', 'sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23'),
  },
  bilingual: {
    label: 'streaming-zipformer-bilingual-zh-en (int8)',
    dir: path.join(__dirname, 'models', 'sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20'),
  },
};

const APP_FIXTURES = path.join(__dirname, '..', 'test', 'fixtures');

function parseArgs() {
  const argv = process.argv.slice(2);
  const get = (flag, def) => {
    const i = argv.indexOf(flag);
    return i >= 0 && argv[i + 1] ? argv[i + 1] : def;
  };
  const model = get('--model', 'zh14m');
  if (!MODELS[model]) {
    console.error(`unknown --model '${model}'; choose from: ${Object.keys(MODELS).join(', ')}`);
    process.exit(1);
  }
  return {
    chunkMs: parseInt(get('--chunk', '100'), 10),
    wav: get('--wav', ''),
    model,
    verbose: argv.includes('--verbose'),
  };
}

function createRecognizer(modelDir) {
  const config = {
    featConfig: { sampleRate: 16000, featureDim: 80 },
    modelConfig: {
      transducer: {
        encoder: path.join(modelDir, 'encoder-epoch-99-avg-1.int8.onnx'),
        decoder: path.join(modelDir, 'decoder-epoch-99-avg-1.onnx'),
        joiner: path.join(modelDir, 'joiner-epoch-99-avg-1.int8.onnx'),
      },
      tokens: path.join(modelDir, 'tokens.txt'),
      numThreads: 2,
      provider: 'cpu',
      debug: 0,
      modelType: 'zipformer',
    },
  };
  return new sherpa.OnlineRecognizer(config);
}

/** Stream the whole wav in chunkMs frames, like the app's AudioWorklet. */
function streamDecode(recognizer, wav, { chunkMs = 100, tailMs = 600, verbose = false } = {}) {
  const chunkSamples = Math.round((16000 * chunkMs) / 1000);
  const stream = recognizer.createStream();
  let decodeMs = 0;
  let lastText = '';
  let firstTextAtMs = -1; // audio-time when the first non-empty partial appeared
  const timeline = [];

  const totalChunks = Math.ceil(wav.samples.length / chunkSamples);
  for (let i = 0; i < totalChunks; i++) {
    const start = i * chunkSamples;
    const end = Math.min(start + chunkSamples, wav.samples.length);
    stream.acceptWaveform({ sampleRate: 16000, samples: wav.samples.subarray(start, end) });
    const d0 = Date.now();
    while (recognizer.isReady(stream)) recognizer.decode(stream);
    decodeMs += Date.now() - d0;
    const audioMs = Math.round((end / 16000) * 1000);
    const text = recognizer.getResult(stream).text;
    if (text !== lastText) {
      if (firstTextAtMs < 0 && text.trim()) firstTextAtMs = audioMs;
      timeline.push({ atMs: audioMs, text });
      lastText = text;
    }
  }

  // tail padding so the last syllables flush out (like VAD trailing silence)
  const tail = new Float32Array(Math.round((16000 * tailMs) / 1000));
  stream.acceptWaveform({ sampleRate: 16000, samples: tail });
  const d1 = Date.now();
  while (recognizer.isReady(stream)) recognizer.decode(stream);
  decodeMs += Date.now() - d1;

  const finalText = recognizer.getResult(stream).text;
  const durMs = (wav.samples.length / 16000) * 1000;

  return { finalText, firstTextAtMs, decodeMs, durMs, timeline };
}

function runOne(recognizer, label, wavPath, args) {
  const wav = sherpa.readWave(wavPath);
  if (wav.sampleRate !== 16000) {
    console.log(`  [skip] ${label}: sampleRate=${wav.sampleRate} (need 16000)`);
    return;
  }
  const r = streamDecode(recognizer, wav, { chunkMs: args.chunkMs, verbose: args.verbose });
  const rtf = r.durMs ? (r.decodeMs / r.durMs).toFixed(3) : '?';
  console.log(`  ${label} (${(r.durMs / 1000).toFixed(2)}s, chunk=${args.chunkMs}ms)`);
  console.log(`    final : ${r.finalText || '(empty)'}`);
  console.log(
    `    first : ${r.firstTextAtMs < 0 ? 'N/A' : r.firstTextAtMs + 'ms (audio-time)'}`,
    `  RTF: ${rtf} (decode ${r.decodeMs}ms)`,
  );
  if (args.verbose) {
    for (const t of r.timeline) {
      console.log(`      @${String(t.atMs).padStart(5)}ms  "${t.text}"`);
    }
  } else {
    // show the first 3 increments so we can eyeball "speaking-while-transcribing"
    const head = r.timeline.slice(0, 3);
    for (const t of head) console.log(`      @${String(t.atMs).padStart(5)}ms  "${t.text}"`);
    if (r.timeline.length > 3) console.log(`      ... (${r.timeline.length} partial updates total)`);
  }
}

function peakRssMb() {
  const m = process.memoryUsage();
  return (m.rss / 1024 / 1024).toFixed(1);
}

function main() {
  const args = parseArgs();
  const modelCfg = MODELS[args.model];
  console.log(`[probe] sherpa-onnx-node v${require('sherpa-onnx-node/package.json').version}`);
  console.log(`[probe] model  : ${modelCfg.label}`);
  console.log(`[probe] chunk  : ${args.chunkMs}ms feed frames`);
  const tLoad = Date.now();
  const recognizer = createRecognizer(modelCfg.dir);
  console.log(`[probe] load   : ${Date.now() - tLoad}ms`);

  const inputs = args.wav
    ? [[args.wav, path.basename(args.wav)]].map(([l, p]) => [l, path.resolve(p)])
    : [
        ['model-test-0.wav', path.join(modelCfg.dir, 'test_wavs', '0.wav')],
        ['model-test-1.wav', path.join(modelCfg.dir, 'test_wavs', '1.wav')],
        ['fixture-zh_16k.wav', path.join(APP_FIXTURES, 'zh_16k.wav')],
        ['fixture-en_test.wav', path.join(APP_FIXTURES, 'en_test.wav')],
      ];

  for (const [label, p] of inputs) {
    try {
      runOne(recognizer, label, p, args);
    } catch (e) {
      console.log(`  ${label}: ERROR ${e.message}`);
    }
  }

  console.log(`[probe] peak RSS : ${peakRssMb()} MB`);
}

main();
