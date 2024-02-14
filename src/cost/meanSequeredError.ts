import * as mj from "mathjs";

export function meanSequeredError(y_true: mj.Matrix, y_pred: mj.Matrix) {
  const sub = mj.subtract(y_true, y_pred);
  const pow = mj.dotMultiply(sub, sub);
  const len = pow.size();
  const add = mj.sum(pow);
  const mean = Number(add) / Number(len[0]);
  return mean;
}

export function dMeanSequeredError(y_true: mj.Matrix, y_pred: mj.Matrix) {
  const sub = mj.subtract(y_pred, y_true);
  return sub;
}
