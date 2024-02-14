import {
  Matrix,
  dot,
  dotDivide,
  dotMultiply,
  log,
  map,
  multiply,
  ones,
  subtract,
} from "mathjs";

export function crossEntropyError(y_true: Matrix, y_pred: Matrix) {
  const logPred = map(y_pred, (val) => log(val));
  // console.log(y_pred);
  const actualPrediksi = dot(y_true, logPred);
  return -Number(actualPrediksi);
}

export function dCrossEntropyError(y_true: Matrix, y_pred: Matrix) {
  const sub = subtract(y_pred, y_true);
  return sub;
}
