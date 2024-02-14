import * as mj from "mathjs";
import calculateActivation from "../activation";
import calculateError from "../cost";
import AdaGrad from "../optimazer/adaGrad";
import { activation } from "../activation/interfaces";
import { cost } from "../cost/interfaces";
import fs from "fs";

export default class RNN {
  WIH: mj.Matrix;
  WHO: mj.Matrix;
  inputNodes: number;
  hiddenNodes: number;
  currentHidden: mj.Matrix;
  seriesDataset: mj.Matrix[][];
  optimazer: AdaGrad[];
  activation: activation;
  cost: cost;
  lr: number = 0.1;
  constructor(
    seriesDataset: mj.Matrix[][],
    inputNodes: number,
    hiddenNodes: number,
    outputNodes: number,
    lr: number,
    activation: activation,
    cost: cost
  ) {
    this.seriesDataset = seriesDataset;
    this.inputNodes = inputNodes;
    this.hiddenNodes = hiddenNodes;
    this.WIH = this.generateMatrix([hiddenNodes, inputNodes + hiddenNodes]);
    this.WHO = this.generateMatrix([outputNodes, hiddenNodes]);
    this.optimazer = [
      new AdaGrad(this.WIH.size()),
      new AdaGrad(this.WHO.size()),
    ];
    this.currentHidden = mj.matrix(mj.zeros([hiddenNodes, 1]));
    this.activation = activation;
    this.cost = cost;
    this.lr = lr;
  }

  saveModels(path: string) {
    const data = {
      WIH: this.WIH.toJSON(),
      WHO: this.WHO.toJSON(),
      // BI: this.BI.toJSON(),
      // BH: this.BH.map((B) => B.toJSON()),
      // BO: this.BO.toJSON(),
    };
    const dataJson = JSON.stringify(data);
    fs.writeFileSync(path, dataJson);
  }

  loadModels(path: string) {
    const dataJson = fs.readFileSync(path, "utf-8");
    const result = JSON.parse(dataJson);
    this.WIH = mj.matrix(result.WIH.data);
    this.WHO = mj.matrix(result.WHO.data);
    // this.BI = mj.matrix(result.BI.data);
    // this.BO = mj.matrix(result.BO.data);
  }

  generateMatrix(shape: number[]): mj.Matrix {
    return mj.matrix(mj.random(shape, -1, 1));
  }

  predict(X_series: mj.Matrix[]): mj.Matrix {
    let pred = mj.matrix();
    for (let X of X_series) {
      let inputs = mj.concat(mj.flatten(X), mj.flatten(this.currentHidden));
      inputs = mj.reshape(inputs, [this.inputNodes + this.hiddenNodes, 1]);
      const [RIH, dIH] = calculateActivation(
        mj.multiply(this.WIH, inputs),
        "tanh"
      );
      const [HO, dHO] = calculateActivation(
        mj.multiply(this.WHO, RIH),
        this.activation
      );
      pred = HO;
      this.currentHidden = RIH;
    }
    return pred;
  }

  train(X: mj.Matrix, y: mj.Matrix) {
    let inputs = mj.concat(mj.flatten(X), mj.flatten(this.currentHidden));
    inputs = mj.reshape(inputs, [this.inputNodes + this.hiddenNodes, 1]);
    const target = y;
    const [RIH, dIH] = calculateActivation(
      mj.multiply(this.WIH, inputs),
      "tanh"
    );
    const [HO, dHO] = calculateActivation(
      mj.multiply(this.WHO, RIH),
      this.activation
    );

    const [loss, err] = calculateError(target, HO, this.cost);

    const d_err_o = mj.dotMultiply(err, dHO);
    const d_err_o_lr = mj.multiply(d_err_o, 1);
    const gHO = mj.multiply(d_err_o_lr, mj.transpose(RIH));
    const newWHO = mj.subtract(
      this.WHO,
      this.optimazer[1].optimazer(this.lr, gHO)
    );

    this.WHO = newWHO;

    const err_ih = mj.multiply(mj.transpose(this.WHO), err);
    const d_err_ih = mj.dotMultiply(err_ih, dIH);
    const d_err_ih_lr = mj.multiply(d_err_ih, 1);
    const gIH = mj.multiply(d_err_ih_lr, mj.transpose(inputs));
    const newWIH = mj.subtract(
      this.WIH,
      this.optimazer[0].optimazer(this.lr, gIH)
    );

    this.WIH = newWIH;
    this.currentHidden = RIH;
    return loss;
  }

  fit(epoch: number, savePath: string = "") {
    let l;
    for (let i = 0; i < epoch; i++) {
      for (let dataset of this.seriesDataset) {
        for (let series in dataset) {
          if (dataset[Number(series) + 1] !== undefined) {
            const loss = this.train(
              dataset[series],
              dataset[Number(series) + 1]
            );
            l = loss;
          }
          console.clear();
          console.log("epoch =>", i);
          console.log("loss =>", l);
        }
        this.currentHidden = mj.matrix(mj.zeros([this.hiddenNodes, 1]));
        if (savePath !== "") this.saveModels(savePath);
      }
    }
  }
}
