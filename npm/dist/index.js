import { loadWasmModule, loadNativeAddon, getSubstrate } from "@tty-pt/stele";

const nativeBinding = loadNativeAddon(import.meta.url, "sepal");
let wasmInstance = null;

export function getSepalWasm() {
  if (wasmInstance) return wasmInstance;
  wasmInstance = loadWasmModule(new URL("./sepal.wasm", import.meta.url));
  return wasmInstance;
}

/**
 * Calculates normalized cosine similarity between two float vectors strictly via WebAssembly
 */
export function cosineSimilarity(vecA, vecB) {
  if (nativeBinding && nativeBinding.cosine) {
    return nativeBinding.cosine(vecA, vecB);
  }

  if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) {
    return 0.0;
  }

  const wasm = getSepalWasm();
  const dim = vecA.length;
  const ptrA = wasm.exports.malloc(dim * 4);
  const ptrB = wasm.exports.malloc(dim * 4);
  const mem = getSubstrate().memory.buffer;
  new Float32Array(mem, ptrA, dim).set(vecA);
  new Float32Array(mem, ptrB, dim).set(vecB);
  const score = wasm.exports.sepal_cosine(ptrA, ptrB, dim);
  wasm.exports.free(ptrA);
  wasm.exports.free(ptrB);
  return score;
}

/**
 * Computes 1-bit sign sketch over float vector strictly via WebAssembly
 */
export function sketchVector(vec) {
  if (!vec || vec.length === 0) return new BigUint64Array(0);
  const wasm = getSepalWasm();
  const dim = vec.length;
  const nwords = Math.ceil(dim / 64);
  const ptrV = wasm.exports.malloc(dim * 4);
  const ptrWords = wasm.exports.malloc(nwords * 8);
  const mem = getSubstrate().memory.buffer;
  new Float32Array(mem, ptrV, dim).set(vec);
  wasm.exports.sepal_wasm_sketch(ptrV, dim, ptrWords, nwords);
  const words = new BigUint64Array(mem, ptrWords, nwords).slice();
  wasm.exports.free(ptrV);
  wasm.exports.free(ptrWords);
  return words;
}

/**
 * Computes bitwise Hamming distance between two 64-bit sketches strictly via WebAssembly
 */
export function hammingDistance(sketchA, sketchB) {
  if (!sketchA || !sketchB || sketchA.length !== sketchB.length || sketchA.length === 0) {
    return 0;
  }
  const wasm = getSepalWasm();
  const nwords = sketchA.length;
  const ptrA = wasm.exports.malloc(nwords * 8);
  const ptrB = wasm.exports.malloc(nwords * 8);
  const mem = getSubstrate().memory.buffer;
  new BigUint64Array(mem, ptrA, nwords).set(sketchA);
  new BigUint64Array(mem, ptrB, nwords).set(sketchB);
  const dist = wasm.exports.sepal_wasm_hamming(ptrA, ptrB, nwords);
  wasm.exports.free(ptrA);
  wasm.exports.free(ptrB);
  return dist;
}

/**
 * VectorIndex: Persistent or in-memory vector index
 */
export class VectorIndex {
  constructor(filePath = "") {
    this.filePath = filePath;
    this.handle = null;
    this._memVectors = new Map();

    if (nativeBinding && nativeBinding.open) {
      this.handle = nativeBinding.open(this.filePath);
    }
  }

  put(ref, vector) {
    if (nativeBinding && this.handle) {
      return nativeBinding.put(this.handle, ref, vector);
    }
    this._memVectors.set(ref, vector);
    return true;
  }

  search(queryVector, k = 10, minScore = 0.0) {
    if (nativeBinding && this.handle) {
      return nativeBinding.search(this.handle, queryVector, k, minScore);
    }

    const scored = [];
    for (const [ref, vec] of this._memVectors.entries()) {
      const sim = cosineSimilarity(queryVector, vec);
      if (sim >= minScore) {
        scored.push({ ref, score: sim });
      }
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, k);
  }
}

/**
 * Pluggable CormAxis implementation for the Corm Recall Kernel
 */
export class SepalVectorAxis {
  constructor(filePath = "") {
    this.name = "vector";
    this.index = new VectorIndex(filePath);
    this._vectors = new Map();
  }

  registerVector(ref, vector) {
    this.index.put(ref, vector);
    this._vectors.set(ref, vector);
  }

  rank(params, ref) {
    const vec = this._vectors.get(ref);
    if (!vec || !params.queryVector) return 0.0;
    return cosineSimilarity(params.queryVector, vec);
  }
}
