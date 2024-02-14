import { Matrix, map } from "mathjs";

export function relu(X: Matrix) {
  return map(X, (x) => {
    if (x > 0) {
      return x;
    } else {
      return 0;
    }
  });
}

export function dRelu(X: Matrix) {
  return map(X, (x) => {
    if (x > 0) {
      return 1;
    } else {
      return 0;
    }
  });
}
