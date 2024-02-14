import * as mj from "mathjs";

export function sigmoid(X: mj.Matrix) {
  return mj.map(X, (x) => {
    const s = 1 / (1 + Math.exp(-x));
    if (s < 0.001) return 0;
    return s;
  });
}

export function dSigmoid(X: mj.Matrix) {
  return mj.map(X, (x) => x * (1 - x));
}
