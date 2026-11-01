export declare function cosineSimilarity(vecA: number[], vecB: number[]): number;

export declare class VectorIndex {
  constructor(filePath?: string);
  put(ref: number, vector: number[]): boolean;
  search(queryVector: number[], k?: number, minScore?: number): Array<{ ref: number; score: number }>;
}

export declare class SepalVectorAxis {
  readonly name: string;
  constructor(filePath?: string);
  registerVector(ref: number, vector: number[]): void;
  rank(params: { queryVector: number[] }, ref: number): number;
}
