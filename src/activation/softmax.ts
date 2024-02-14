import { Matrix } from "mathjs";
import * as mj from "mathjs";

export function softmax(X: Matrix): Matrix {
  const sumExp = mj.map(X, (x) => mj.exp(x));
  const dot = mj.dot(mj.ones(sumExp.size()), sumExp);
  const exp = mj.map(X, (x) => {
    const a = mj.exp(x) / Number(dot);
    if (a < 0.01) return 0;
    return a;
  });
  return exp;
}

export function dSoftmax(X: Matrix): Matrix {
  const min = mj.subtract(1, X);
  const mul = mj.dotMultiply(X, min);
  return mul;
}
