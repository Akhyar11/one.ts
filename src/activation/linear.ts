import { Matrix, map, max } from "mathjs";

export function linear(X: Matrix) {
  return map(X, (x) => max(x));
}

export function dLinear(X: Matrix) {
  return map(X, () => 1);
}
