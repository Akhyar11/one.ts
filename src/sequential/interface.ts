import { optimazerType } from "../optimazer/interfaces";

export interface config {
  err: cost;
  optimazer: optimazerType;
}

export type activation = "sigmoid" | "linear" | "tanh" | "relu" | "lRelu";
export type cost = "mse" | "crossEntropy";
