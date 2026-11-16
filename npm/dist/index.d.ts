export declare function cosineSimilarity(vecA: number[] | Float32Array, vecB: number[] | Float32Array, options?: { wasm?: boolean }): number;
export declare function sketchVector(vec: number[] | Float32Array, options?: { wasm?: boolean }): BigUint64Array;
export declare function hammingDistance(sketchA: BigUint64Array, sketchB: BigUint64Array, options?: { wasm?: boolean }): number;
export declare function configureEmbeddings(url: string, model: string, apiKey?: string): boolean;

export declare class VectorIndex {
  constructor(filePath?: string);
  put(ref: number, vector: number[] | Float32Array): boolean;
  get(ref: number): Float32Array | null;
  del(ref: number): boolean;
  count(): number;
  search(queryVector: number[] | Float32Array, k?: number, minScore?: number, m?: number): Array<{ ref: number; score: number }>;
  fillApprox(queryVector: number[] | Float32Array, m?: number, minScore?: number): number[];
  rank(queryVector: number[] | Float32Array, ref: number, minScore?: number): number;
  storeTyped(ref: number, uint8Array: Uint8Array, qtype?: number): boolean;
  close(): void;
}

export declare class SepalVectorAxis {
  readonly name: string;
  constructor(filePath?: string);
  registerVector(ref: number, vector: number[] | Float32Array): void;
  unstore(ref: number): boolean;
  storeTyped(ref: number, uint8Array: Uint8Array, qtype?: number): boolean;
  fill(params: { queryVector: number[] | Float32Array; m?: number; minScore?: number }): number[];
  rank(params: { queryVector: number[] | Float32Array; minScore?: number }, ref: number): number;
  close(): void;
}
