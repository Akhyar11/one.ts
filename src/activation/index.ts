import * as mj from "mathjs";
import { activation } from "./interfaces";
import { dSigmoid, sigmoid } from "./sigmoid";
import { dTanh, tanh } from "./tanh";
import { dRelu, relu } from "./reru";
import { lRelu } from "./lRelu";
import { dLinear, linear } from "./linear";
import { dSoftmax, softmax } from "./softmax";
import { dLogSoftmax, logSoftmax } from "./logSoftmax";

function calculateActivation(
  X: mj.Matrix,
  activation: activation
): [mj.Matrix, mj.Matrix] {
  let result: mj.Matrix;
  let dResult: mj.Matrix;
  switch (activation) {
    case "sigmoid":
      result = sigmoid(X);
      dResult = dSigmoid(result);
      break;
    case "tanh":
      result = tanh(X);
      dResult = dTanh(result);
      break;
    case "relu":
      result = relu(X);
      dResult = dRelu(result);
      break;
    case "lRelu":
      result = lRelu(X);
      dResult = dRelu(result);
      break;
    case "softmax":
      result = softmax(X);
      dResult = dSoftmax(result);
      break;
    case "logSoftmax":
      result = logSoftmax(X);
      dResult = dLogSoftmax(result);
      break;
    default:
      result = linear(X);
      dResult = dLinear(result);
      break;
  }
  return [result, dResult];
}

export default calculateActivation;
