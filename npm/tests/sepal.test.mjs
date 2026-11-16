import test from "node:test";
import assert from "node:assert/strict";
import {
  cosineSimilarity,
  sketchVector,
  hammingDistance,
  VectorIndex,
  SepalVectorAxis,
} from "../dist/index.js";

test("cosineSimilarity calculates identical and orthogonal vectors (array & Float32Array)", () => {
  const v1 = new Float32Array([1.0, 0.0, 0.0]);
  const v2 = [1.0, 0.0, 0.0];
  const v3 = new Float32Array([0.0, 1.0, 0.0]);

  assert.ok(Math.abs(cosineSimilarity(v1, v2) - 1.0) < 0.001);
  assert.ok(Math.abs(cosineSimilarity(v1, v3) - 0.0) < 0.001);
});

test("sketchVector and hammingDistance", () => {
  const s1 = sketchVector(new Float32Array([1, 1, 1, 1]));
  const s2 = sketchVector(new Float32Array([1, 1, 1, 1]));
  const s3 = sketchVector(new Float32Array([-1, -1, 1, 1]));

  assert.equal(hammingDistance(s1, s2), 0);
  assert.equal(hammingDistance(s1, s3), 2);
});

test("VectorIndex put, get, count, del, and search ranking with Float32Array", () => {
  const idx = new VectorIndex();
  idx.put(1, new Float32Array([0.1, 0.9, 0.0]));
  idx.put(2, [0.8, 0.1, 0.0]);
  idx.put(3, new Float32Array([0.0, 0.85, 0.1]));

  assert.equal(idx.count(), 3);
  const v1 = idx.get(1);
  assert.ok(v1 instanceof Float32Array);
  assert.equal(v1.length, 3);

  const query = new Float32Array([0.0, 1.0, 0.0]);
  const hits = idx.search(query, 2, 0.5);
  assert.ok(hits.length >= 1);
  assert.ok(hits[0].ref === 1 || hits[0].ref === 3);

  assert.equal(idx.del(1), true);
  assert.equal(idx.get(1), null);
  assert.equal(idx.count(), 2);
  idx.close();
});

test("SepalVectorAxis fill, rank, and unstore", () => {
  const axis = new SepalVectorAxis();
  axis.registerVector(10, new Float32Array([1.0, 0.0, 0.0]));
  axis.registerVector(20, new Float32Array([0.0, 1.0, 0.0]));

  const q = new Float32Array([1.0, 0.0, 0.0]);
  const filled = axis.fill({ queryVector: q, m: 10 });
  assert.ok(filled.includes(10));

  const r10 = axis.rank({ queryVector: q }, 10);
  const r20 = axis.rank({ queryVector: q }, 20);
  assert.ok(r10 > 0.99);
  assert.ok(Math.abs(r20) < 0.01);

  assert.equal(axis.unstore(10), true);
  axis.close();
});
