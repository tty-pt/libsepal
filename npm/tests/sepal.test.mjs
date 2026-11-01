import test from "node:test";
import assert from "node:assert/strict";
import { cosineSimilarity, VectorIndex } from "../dist/index.js";

test("cosineSimilarity calculates identical and orthogonal vectors", () => {
  const v1 = [1.0, 0.0, 0.0];
  const v2 = [1.0, 0.0, 0.0];
  const v3 = [0.0, 1.0, 0.0];

  assert.ok(Math.abs(cosineSimilarity(v1, v2) - 1.0) < 0.001);
  assert.ok(Math.abs(cosineSimilarity(v1, v3) - 0.0) < 0.001);
});

test("VectorIndex put and search ranking", () => {
  const idx = new VectorIndex();
  idx.put(1, [0.1, 0.9, 0.0]);
  idx.put(2, [0.8, 0.1, 0.0]);
  idx.put(3, [0.0, 0.85, 0.1]);

  const query = [0.0, 1.0, 0.0];
  const hits = idx.search(query, 2, 0.5);
  assert.ok(hits.length >= 1);
  assert.ok(hits[0].ref === 1 || hits[0].ref === 3);
});
