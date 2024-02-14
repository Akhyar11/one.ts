import { Matrix } from "mathjs";
import * as mj from "mathjs";
import { softmax } from "./softmax";
import clip from "../utils/clip";

export function logSoftmax(X: Matrix): Matrix {
  const soft = softmax(X);
  const log = mj.map(soft, (val) => mj.log(val));
  return log;
}

export function dLogSoftmax(X: Matrix): Matrix {
  const d = mj.map(X, (x) => mj.exp(x));
  return d;
}
