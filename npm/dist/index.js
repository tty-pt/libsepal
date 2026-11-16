import { loadWasmModule, loadNativeAddon, getSubstrate } from "@tty-pt/stele";

const nativeBinding = loadNativeAddon(import.meta.url, "sepal");
let wasmInstance = null;

export function getSepalWasm() {
  if (wasmInstance) return wasmInstance;
  wasmInstance = loadWasmModule(new URL("./sepal.wasm", import.meta.url));
  return wasmInstance;
}

/**
 * Calculates normalized cosine similarity between two float vectors
 */
export function cosineSimilarity(vecA, vecB, options = {}) {
  if (!vecA || !vecB || vecA.length !== vecB.length || vecA.length === 0) {
    return 0.0;
  }
  if (!options.wasm && nativeBinding && nativeBinding.cosine) {
    return nativeBinding.cosine(vecA, vecB);
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
 * Computes 1-bit sign sketch over float vector
 */
export function sketchVector(vec, options = {}) {
  if (!vec || vec.length === 0) return new BigUint64Array(0);
  if (!options.wasm && nativeBinding && nativeBinding.sketch) {
    return nativeBinding.sketch(vec);
  }
  const wasm = getSepalWasm();
  const dim = vec.length;
  const nwords = Math.ceil(dim / 64);
  const ptrV = wasm.exports.malloc(dim * 4);
  const ptrWords = wasm.exports.malloc(nwords * 8);
  new Float32Array(getSubstrate().memory.buffer, ptrV, dim).set(vec);
  wasm.exports.sepal_wasm_sketch(ptrV, dim, ptrWords, nwords);
  const words = new BigUint64Array(getSubstrate().memory.buffer, ptrWords, nwords).slice();
  wasm.exports.free(ptrV);
  wasm.exports.free(ptrWords);
  return words;
}

/**
 * Computes bitwise Hamming distance between two 64-bit sketches
 */
export function hammingDistance(sketchA, sketchB, options = {}) {
  if (!sketchA || !sketchB || sketchA.length !== sketchB.length || sketchA.length === 0) {
    return 0;
  }
  if (!options.wasm && nativeBinding && nativeBinding.hamming) {
    return nativeBinding.hamming(sketchA, sketchB);
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

export function configureEmbeddings(url, model, apiKey = "") {
  if (!nativeBinding || !nativeBinding.configureEmbeddings) {
    throw new Error("@tty-pt/sepal native addon is required for configureEmbeddings");
  }
  return nativeBinding.configureEmbeddings(url || "", model || "", apiKey || "");
}

/**
 * VectorIndex: Persistent or in-memory vector store backed strictly by libsepal C engine
 */
export class VectorIndex {
  constructor(filePath = "") {
    if (!nativeBinding || !nativeBinding.open) {
      throw new Error("@tty-pt/sepal native addon is required for VectorIndex");
    }
    this.filePath = filePath;
    this.handle = nativeBinding.open(this.filePath);
  }

  put(ref, vector) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.put(this.handle, Number(ref), vector);
  }

  get(ref) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.get(this.handle, Number(ref));
  }

  del(ref) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.del(this.handle, Number(ref));
  }

  count() {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.count(this.handle);
  }

  search(queryVector, k = 10, minScore = 0.0, m = 0) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.search(this.handle, queryVector, k, minScore, m);
  }

  fillApprox(queryVector, m = 64, minScore = -1.0) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.fillApprox(this.handle, queryVector, m, minScore);
  }

  rank(queryVector, ref, minScore = -1.0) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.rank(this.handle, queryVector, Number(ref), minScore);
  }

  storeTyped(ref, uint8Array, qtype = 0) {
    if (!this.handle) throw new Error("VectorIndex is closed");
    return nativeBinding.storeTyped(this.handle, Number(ref), uint8Array, qtype);
  }

  close() {
    if (this.handle) {
      nativeBinding.close(this.handle);
      this.handle = null;
    }
  }
}

/**
 * Pluggable CormAxis implementation for the Corm Recall Kernel backed strictly by libsepal C engine
 */
export class SepalVectorAxis {
  constructor(filePath = "") {
    this.name = "vector";
    this.index = new VectorIndex(filePath);
  }

  registerVector(ref, vector) {
    this.index.put(ref, vector);
  }

  unstore(ref) {
    return this.index.del(ref);
  }

  storeTyped(ref, uint8Array, qtype = 0) {
    return this.index.storeTyped(ref, uint8Array, qtype);
  }

  fill(params) {
    if (!params?.queryVector) return [];
    const m = params.m ?? 64;
    const minScore = params.minScore ?? -1.0;
    return this.index.fillApprox(params.queryVector, m, minScore);
  }

  rank(params, ref) {
    if (!params?.queryVector) return 0.0;
    const minScore = params.minScore ?? -1.0;
    return this.index.rank(params.queryVector, ref, minScore);
  }

  close() {
    this.index.close();
  }
}
