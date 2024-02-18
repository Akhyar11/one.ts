import * as mj from "mathjs";
import calculateActivation from "../activation";
import calculateError from "../cost";
import AdaGrad from "../optimazer/adaGrad";
import Sgd from "../optimazer/sgd";
import { activation } from "../activation/interfaces";
import { cost } from "../cost/interfaces";
import fs from "fs";
import { optimazer, optimazerType } from "../optimazer/interfaces";

export default class RNN {
  WIH: mj.Matrix;
  WHH: mj.Matrix;
  WHO: mj.Matrix;
  BI: mj.Matrix;
  BO: mj.Matrix;
  activation: activation = "lRelu";
  lr: number;
  private optimazerIH: optimazer = new Sgd();
  private optimazerHH: optimazer = new Sgd();
  private optimazerHO: optimazer = new Sgd();
  private optimazerBI: optimazer = new Sgd();
  private optimazerBO: optimazer = new Sgd();
  private prev_hidden: mj.Matrix;
  private outputs: mj.Matrix[] = [];
  private dOutputs: mj.Matrix[] = [];
  private hiddens: mj.Matrix[] = [];
  private dHiddens: mj.Matrix[] = [];
  private next_hidden: mj.Matrix = mj.matrix();
  constructor(
    inputNodes: number,
    hiddenNodes: number,
    outputNodes: number,
    activation: activation,
    optimazer: optimazerType
  ) {
    this.WIH = mj.matrix(mj.random([hiddenNodes, inputNodes], -1, 1));
    this.WHH = mj.matrix(mj.random([hiddenNodes, hiddenNodes], -1, 1));
    this.WHO = mj.matrix(mj.random([outputNodes, hiddenNodes], -1, 1));
    this.BI = mj.matrix(mj.zeros([hiddenNodes, 1]));
    this.BO = mj.matrix(mj.zeros([outputNodes, 1]));
    this.prev_hidden = mj.matrix(mj.zeros([hiddenNodes, 1]));
    this.next_hidden = mj.matrix(mj.zeros([hiddenNodes, 1]));
    this.lr = 0.1;
    this.activation = activation;
    if (optimazer === "adaGrad") {
      this.optimazerIH = new AdaGrad([hiddenNodes, inputNodes]);
      this.optimazerHH = new AdaGrad([hiddenNodes, hiddenNodes]);
      this.optimazerHO = new AdaGrad([outputNodes, hiddenNodes]);
      this.optimazerBI = new AdaGrad([hiddenNodes, 1]);
      this.optimazerBO = new AdaGrad([outputNodes, 1]);
    } else {
      this.optimazerIH = new Sgd();
      this.optimazerHH = new Sgd();
      this.optimazerHO = new Sgd();
      this.optimazerBI = new Sgd();
      this.optimazerBO = new Sgd();
    }
  }

  predict(x: mj.Matrix[]) {
    let xo: mj.Matrix = mj.matrix();
    this.prev_hidden = mj.matrix(mj.zeros([this.WHH.size()[0], 1]));
    for (let i in x) {
      let xi = mj.multiply(this.WIH, x[i]);
      let xh = mj.add(
        mj.add(xi, mj.multiply(this.WHH, this.prev_hidden)),
        this.BI
      );
      [xh] = calculateActivation(xh, "tanh");
      this.prev_hidden = xh;

      [xo] = calculateActivation(
        mj.add(mj.multiply(this.WHO, xh), this.BO),
        this.activation
      );
    }

    return xo;
  }

  train(x: mj.Matrix[], y: mj.Matrix[]) {
    let dXh = mj.matrix();
    for (let i in x) {
      let xi = mj.multiply(this.WIH, x[i]);
      let xh = mj.add(
        mj.add(xi, mj.multiply(this.WHH, this.prev_hidden)),
        this.BI
      );
      [xh, dXh] = calculateActivation(xh, "tanh");
      this.prev_hidden = xh;
      this.hiddens[i] = xh;
      this.dHiddens[i] = dXh;

      let [xo, dHO] = calculateActivation(
        mj.add(mj.multiply(this.WHO, xh), this.BO),
        this.activation
      );
      this.outputs[i] = xo;
      this.dOutputs[i] = dHO;
    }

    let [loss]: [mj.MathNumericType, mj.MathType] = [0, mj.matrix()];
    let sumErrBO = mj.matrix(mj.zeros(this.outputs[0].size()));
    let sumErrBI = mj.matrix(mj.zeros(this.hiddens[0].size()));
    let gWHO = mj.matrix(mj.zeros(this.WHO.size()));
    let gWHH = mj.matrix(mj.zeros(this.WHH.size()));
    let gWIH = mj.matrix(mj.zeros(this.WIH.size()));
    let indexBO = 0;
    let indexBI = 0;

    for (let i in y) {
      let [l, err] = calculateError(y[i], this.outputs[i], "mse");
      loss = mj.add(loss, l);
      err = mj.dotMultiply(err, this.dOutputs[i]);
      sumErrBO = mj.add(sumErrBO, err);
      gWHO = mj.add(gWHO, mj.multiply(err, mj.transpose(this.hiddens[i])));
      let gO = mj.multiply(mj.transpose(this.WHO), err);
      let gH = mj.add(
        gO,
        mj.multiply(mj.transpose(this.WHH), this.next_hidden)
      );
      gH = mj.dotMultiply(gH, this.dHiddens[i]);
      this.next_hidden = gH;

      if (Number(i) > 0) {
        gWHH = mj.add(
          gWHH,
          mj.multiply(gH, mj.transpose(this.hiddens[Number(i) - 1]))
        );
        sumErrBI = mj.add(sumErrBI, gH);
        indexBI = Number(i);
      }

      gWIH = mj.add(gWIH, mj.multiply(gH, mj.transpose(x[i])));

      indexBO = Number(i) + 1;
    }
    sumErrBO = mj.dotDivide(sumErrBO, indexBO);
    sumErrBI = mj.dotDivide(sumErrBI, indexBI);
    let gBO = this.optimazerBO.optimazer(this.lr, sumErrBO);
    let gBI = this.optimazerBI.optimazer(this.lr, sumErrBI);
    gWHO = this.optimazerHO.optimazer(this.lr, gWHO);
    gWHH = this.optimazerHH.optimazer(this.lr, gWHH);
    gWIH = this.optimazerIH.optimazer(this.lr, gWIH);

    this.WHO = mj.subtract(this.WHO, gWHO);
    this.WHH = mj.subtract(this.WHH, gWHH);
    this.WIH = mj.subtract(this.WIH, gWIH);
    this.BO = mj.subtract(this.BO, gBO);
    this.BI = mj.subtract(this.BI, gBI);

    return mj.dotDivide(loss, indexBO);
  }

  fit(X: mj.Matrix[][], y: mj.Matrix[][], epochs: number, path: string = "") {
    for (let i = 0; i < epochs; i++) {
      for (let j in X) {
        const loss = this.train(X[j], y[j]);
        path === "" ? null : this.saveModels(path);
        console.clear();
        console.log("epoch =>", i);
        console.log("loss =>", loss);
      }
    }
  }
  saveModels(path: string) {
    const data = {
      WIH: this.WIH.toJSON(),
      WHO: this.WHO.toJSON(),
      WHH: this.WHH.toJSON(),
      BI: this.BI.toJSON(),
      BO: this.BO.toJSON(),
    };
    const dataJson = JSON.stringify(data);
    fs.writeFileSync(path, dataJson);
  }
  loadModels(path: string) {
    const dataJson = fs.readFileSync(path, "utf-8");
    const result = JSON.parse(dataJson);
    this.WIH = mj.matrix(result.WIH.data);
    this.WHO = mj.matrix(result.WHO.data);
    this.WHH = mj.matrix(result.WHH.data);
    this.BI = mj.matrix(result.BI.data);
    this.BO = mj.matrix(result.BO.data);
  }
}
