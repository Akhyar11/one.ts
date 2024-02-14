import { Matrix, map } from "mathjs";

export function lRelu(X: Matrix) {
  return map(X, (x) => {
    if (x > 0) {
      return x;
    } else {
      return 0.01 * x;
    }
  });
}

export function dLRelu(X: Matrix) {
  return map(X, (x) => {
    if (x > 0) {
      return 1;
    } else {
      return 0;
    }
  });
}
