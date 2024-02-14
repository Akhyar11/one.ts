import * as mj from "mathjs";

export function tanh(X: mj.Matrix) {
  return mj.map(X, mj.tanh);
}

export function dTanh(X: mj.Matrix) {
  const sub = mj.map(X, (x) => 1 - Math.pow(x, 2));
  return sub;
}
