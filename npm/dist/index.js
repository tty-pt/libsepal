import { createRequire } from "node:module";
import path from "node:path";

let nativeBinding = null;
try {
  const req = createRequire(import.meta.url);
  nativeBinding = req("../build/Release/sepal_node.node");
} catch {
  try {
    const req = createRequire(import.meta.url);
    nativeBinding = req(path.join(process.cwd(), "build/Release/sepal_node.node"));
  } catch {
    // Fallback to pure JS
  }
}

function getSharedWasm() {
  if (globalThis.__ttypt_shared_wasm__) {
    return globalThis.__ttypt_shared_wasm__;
  }
  try {
    const req = createRequire(import.meta.url);
    const qsysPaths = [
      path.join(path.dirname(import.meta.url.replace("file://", "")), "../../libqsys/npm/dist/index.js"),
      path.join(process.cwd(), "node_modules/@ttypt/qsys/dist/index.js"),
      "@ttypt/qsys"
    ];
    for (const p of qsysPaths) {
      try {
        const mod = req(p);
        if (mod && mod.getSharedWasm) {
          return mod.getSharedWasm();
        }
      } catch {}
    }
  } catch {}
  return null;
}

/**
 * Calculates normalized cosine similarity between two float vectors
 */
export function cosineSimilarity(vecA, vecB) {
  if (nativeBinding && nativeBinding.cosine) {
    return nativeBinding.cosine(vecA, vecB);
  }

  const wasm = getSharedWasm();
  if (wasm && vecA && vecB && vecA.length === vecB.length && vecA.length > 0) {
    const dim = vecA.length;
    const ptrA = wasm.malloc(dim * 4);
    const ptrB = wasm.malloc(dim * 4);
    new Float32Array(wasm.memory.buffer, ptrA, dim).set(vecA);
    new Float32Array(wasm.memory.buffer, ptrB, dim).set(vecB);
    const score = wasm.sepal_cosine(ptrA, ptrB, dim);
    wasm.free(ptrA);
    wasm.free(ptrB);
    return score;
  }

  // Pure JS fallback
  if (!vecA || !vecB || vecA.length !== vecB.length) return 0.0;
  let dot = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0.0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
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
