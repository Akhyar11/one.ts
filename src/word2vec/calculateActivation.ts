import * as mj from "mathjs";
import { activation } from "../activation/interfaces";
import { dSigmoid, sigmoid } from "../activation/sigmoid";
import { dTanh, tanh } from "../activation/tanh";
import { dRelu, relu } from "../activation/reru";
import { dLRelu, lRelu } from "../activation/lRelu";
import { dSoftmax, softmax } from "../activation/softmax";
import { dLogSoftmax, logSoftmax } from "../activation/logSoftmax";
import { dLinear, linear } from "../activation/linear";

function calculateActivation(
  X: mj.Matrix,
  activation: activation
): [mj.Matrix, mj.Matrix] {
  let result: mj.Matrix = X;
  let dResult: mj.Matrix = mj.matrix();
  switch (activation) {
    case "sigmoid":
      result = sigmoid(result);
      dResult = dSigmoid(result);
      break;
    case "tanh":
      result = tanh(result);
      dResult = dTanh(result);
      break;
    case "relu":
      result = relu(result);
      dResult = dRelu(result);
      break;
    case "lRelu":
      result = lRelu(result);
      dResult = dLRelu(result);
      break;
    case "softmax":
      result = softmax(result);
      dResult = dSoftmax(result);
      break;
    case "logSoftmax":
      result = logSoftmax(result);
      dResult = dLogSoftmax(result);
      break;
    default:
      result = linear(result);
      result = dLinear(result);
      break;
  }
  return [result, dResult];
}

export default calculateActivation;
