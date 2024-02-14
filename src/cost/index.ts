import * as mj from "mathjs";
import { cost } from "./interfaces";
import { crossEntropyError, dCrossEntropyError } from "./crossEntropyError";
import { dMeanSequeredError, meanSequeredError } from "./meanSequeredError";

function calculateError(
  y_true: mj.Matrix,
  y_pred: mj.Matrix,
  cost: cost
): [mj.MathNumericType | number, mj.MathType] {
  let loss;
  let err;

  switch (cost) {
    case "crossEntropy":
      err = dCrossEntropyError(y_true, y_pred);
      loss = crossEntropyError(y_true, y_pred);
      break;
    default:
      err = dMeanSequeredError(y_true, y_pred);
      loss = meanSequeredError(y_true, y_pred);
      break;
  }

  return [loss, err];
}

export default calculateError;
