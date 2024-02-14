import * as mj from "mathjs";
import { config } from "./interface";
import { activation } from "../activation/interfaces";
import fs, { readFileSync } from "fs";
import calculateActivation from "../activation";
import calculateError from "../cost";
import { optimazer } from "../optimazer/interfaces";
import AdaGrad from "../optimazer/adaGrad";
import Sgd from "../optimazer/sgd";

export default class Sequential {
  config: config;
  lr: number;
  WIH: mj.Matrix = mj.matrix();
  WHH: mj.Matrix[] = [mj.matrix()];
  WHO: mj.Matrix = mj.matrix();
  BI: mj.Matrix = mj.matrix();
  BH: mj.Matrix[] = [mj.matrix()];
  BO: mj.Matrix = mj.matrix();
  inputs: mj.Matrix[] = [];
  hiddens: mj.Matrix[][] = [];
  hiddenInput: mj.Matrix[] = [];
  outputs: mj.Matrix[] = [];
  target: mj.Matrix = this.inputs[0];
  private optimazerWIH: optimazer = new Sgd();
  private optimazerWHH: optimazer[] = [new Sgd()];
  private optimazerWHO: optimazer = new Sgd();
  private optimazerBI: optimazer = new Sgd();
  private optimazerBH: optimazer[] = [new Sgd()];
  private optimazerBO: optimazer = new Sgd();
  private updateErr: mj.MathType = mj.matrix();
  private loss: mj.MathNumericType | number = 0;
  inputNodes: number = 0;
  hiddenNodes: [number, activation][] = [];
  outputNodes: [number, activation] = [0, "linear"];
  private calculateActivation: Function;
  private calculateError: Function;
  constructor(config: config) {
    this.config = config;
    this.lr = 0.1;
    this.calculateActivation = calculateActivation;
    this.calculateError = calculateError;
  }

  saveModels(path: string) {
    const data = {
      WIH: this.WIH.toJSON(),
      WHH: this.WHH.map((W) => W.toJSON()),
      WHO: this.WHO.toJSON(),
      BI: this.BI.toJSON(),
      BH: this.BH.map((B) => B.toJSON()),
      BO: this.BO.toJSON(),
    };
    const dataJson = JSON.stringify(data);
    fs.writeFileSync(path, dataJson);
  }

  loadModels(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const result = JSON.parse(dataJson);
    this.WIH = mj.matrix(result.WIH.data);
    this.WHO = mj.matrix(result.WHO.data);
    for (let i in result.WHH) {
      this.WHH[Number(i)] = mj.matrix(result.WHH[Number(i)].data);
    }
    this.BI = mj.matrix(result.BI.data);
    this.BO = mj.matrix(result.BO.data);
    for (let i in result.BH) {
      this.BH[Number(i)] = mj.matrix(result.BH[Number(i)].data);
    }
  }

  info() {
    console.log("========== INFO ==========\n");
    console.log("Input Nodes =>", this.inputNodes);
    console.log("Hidden Nodes =>", this.hiddenNodes);
    console.log("Output Nodes =>", this.outputNodes);
    console.log("Error =>", this.loss.toString());
    console.log("Learning Rate =>", this.lr);
    console.log("Weight Input-Hidden =>", this.WIH.toArray());
    for (let WHH of this.WHH) {
      console.log("Weight Hidden-Hidden =>", WHH.toArray());
    }
    console.log("Weight Hidden-Output =>", this.WHO.toArray());
    console.log("\n========== END INFO ==========\n");
  }

  inputLayer(inputNodes: number): void {
    this.inputNodes = inputNodes;
  }

  hiddenLayer(hiddenNodes: number, activation: activation): void {
    if (this.hiddenNodes[0] === undefined) {
      this.hiddenNodes.push([hiddenNodes, activation]);
      this.WIH = this.generateMatrix([this.hiddenNodes[0][0], this.inputNodes]);
      this.BI = mj.matrix(mj.zeros([this.hiddenNodes[0][0], 1]));
      this.optimazerWIH = this.choiseOptimazer([
        this.hiddenNodes[0][0],
        this.inputNodes,
      ]);
      this.optimazerBI = this.choiseOptimazer([this.hiddenNodes[0][0], 1]);
    } else {
      this.hiddenNodes.push([hiddenNodes, activation]);
      if (this.hiddenNodes.length !== 1) {
        for (let nodes = 0; nodes < this.hiddenNodes.length; nodes++) {
          if (nodes + 1 !== this.hiddenNodes.length) {
            this.WHH[nodes] = this.generateMatrix([
              this.hiddenNodes[nodes + 1][0],
              this.hiddenNodes[nodes][0],
            ]);
            this.BH[nodes] = mj.matrix(
              mj.zeros([this.hiddenNodes[nodes + 1][0], 1])
            );
            this.optimazerWHH[nodes] = this.choiseOptimazer([
              this.hiddenNodes[nodes + 1][0],
              this.hiddenNodes[nodes][0],
            ]);
            this.optimazerBH[nodes] = this.choiseOptimazer([
              this.hiddenNodes[nodes + 1][0],
              1,
            ]);
          }
        }
      }
    }
  }

  hiddenAsOutputLayer(hiddenNodes: number): mj.Matrix {
    for (let nodes in this.hiddenNodes) {
      if (hiddenNodes === 0) {
        this.calculateInputLayer();
        return this.hiddenInput[0];
      } else if (
        Number(nodes) === hiddenNodes &&
        hiddenNodes + 1 !== this.hiddenNodes.length
      ) {
        this.calculateInputLayer();
        this.calculateHiddenLayer();
        return this.hiddens[Number(nodes) - 1][0];
      } else if (Number(nodes) + 1 === this.hiddenNodes.length) {
        this.calculateInputLayer();
        this.calculateHiddenLayer();
        return this.hiddenInput[0];
      }
    }
    return mj.matrix();
  }

  hiddenAsInputLayer(hiddenNodes: number, input: mj.Matrix) {
    const hidden = [input];
    let hiddenLayer = mj.matrix();
    for (let nodes in this.hiddenNodes) {
      if (hiddenNodes === 0) {
        if (this.hiddenNodes.length !== 1) {
          for (let i = 0; i < this.WHH.length; i++) {
            hiddenLayer = mj.add(
              mj.multiply(this.WHH[i], hidden[i]),
              this.BH[i]
            );
            const [result] = this.calculateActivation(
              hiddenLayer,
              this.hiddenNodes[i + 1][1]
            );
            hidden[i + 1] = result;
          }
        }
        hiddenLayer = mj.add(mj.multiply(this.WHO, hidden[0]), this.BO);
        const [output] = this.calculateActivation(
          hiddenLayer,
          this.outputNodes[1]
        );
        return output;
      } else if (
        Number(nodes) === hiddenNodes &&
        hiddenNodes + 1 !== this.hiddenNodes.length
      ) {
        for (let i = hiddenNodes; i < this.WHH.length; i++) {
          hiddenLayer = mj.add(
            mj.multiply(this.WHH[i], hidden[i - hiddenNodes]),
            this.BH[i]
          );
          const [result] = this.calculateActivation(
            hiddenLayer,
            this.hiddenNodes[i][1]
          );
          hidden[i - hiddenNodes + 1] = result;
        }
        const output = this.calculateOutputLayer();
        return output;
      } else if (hiddenNodes + 1 === this.hiddenNodes.length) {
        const [output] = this.calculateActivation(
          hidden[0],
          this.WHO,
          this.outputNodes[1]
        );
        return output;
      }
    }

    const output = this.calculateOutputLayer();
    return output;
  }

  outputLayer(outputNodes: number, activation: activation): void {
    const len = this.hiddenNodes.length;
    this.outputNodes = [outputNodes, activation];
    this.WHO = this.generateMatrix([
      this.outputNodes[0],
      this.hiddenNodes[len - 1][0],
    ]);
    this.BO = mj.matrix(mj.zeros(this.outputNodes[0], 1));
    this.optimazerWHO = this.choiseOptimazer([
      this.outputNodes[0],
      this.hiddenNodes[len - 1][0],
    ]);
    this.optimazerBO = this.choiseOptimazer([this.outputNodes[0], 1]);
  }

  generateMatrix(shape: number[]) {
    return mj.matrix(mj.random(shape, -1, 1));
  }

  calculateInputLayer() {
    const inputLayer = mj.add(mj.multiply(this.WIH, this.inputs[0]), this.BI);
    const [hidden, dHidden] = this.calculateActivation(
      inputLayer,
      this.hiddenNodes[0][1]
    );
    this.hiddenInput = [hidden, dHidden];
  }

  calculateHiddenLayer() {
    this.hiddens = [];
    let hidenLeyer = mj.matrix();
    if (this.hiddenNodes.length !== 1) {
      for (let nodes = 0; nodes < this.hiddenNodes.length; nodes++) {
        if (nodes + 1 !== this.hiddenNodes.length) {
          if (this.hiddens[0]) {
            hidenLeyer = mj.add(
              mj.multiply(this.WHH[nodes], this.hiddens[nodes - 1][0]),
              this.BH[nodes]
            );
            const [hidden, dHidden] = this.calculateActivation(
              hidenLeyer,
              this.hiddenNodes[nodes + 1][1]
            );
            this.hiddens[nodes] = [hidden, dHidden];
          } else {
            hidenLeyer = mj.add(
              mj.multiply(this.WHH[nodes], this.hiddenInput[0]),
              this.BH[nodes]
            );
            const [hidden, dHidden] = this.calculateActivation(
              hidenLeyer,
              this.hiddenNodes[nodes + 1][1]
            );
            this.hiddens[nodes] = [hidden, dHidden];
          }
        }
      }
    }
  }

  calculateOutputLayer() {
    const len = this.hiddens.length;
    let outputLayer = mj.matrix();
    if (this.hiddenNodes.length !== 1) {
      outputLayer = mj.add(
        mj.multiply(this.WHO, this.hiddens[len - 1][0]),
        this.BO
      );
      const [output, dOutput] = this.calculateActivation(
        outputLayer,
        this.outputNodes[1]
      );
      this.outputs = [output, dOutput];
      return output;
    } else {
      outputLayer = mj.add(mj.multiply(this.WHO, this.hiddenInput[0]), this.BO);
      const [output, dOutput] = this.calculateActivation(
        outputLayer,
        this.outputNodes[1]
      );
      this.outputs = [output, dOutput];
      return output;
    }
  }

  updateWOutputLayer() {
    const [loss, err] = this.calculateError(
      this.target,
      this.outputs[0],
      this.config.err
    );
    this.updateErr = err;
    if (this.hiddenNodes.length !== 1) {
      const len = this.hiddens.length;
      const newW = this.updateWightOutput(
        this.WHO,
        this.BO,
        this.optimazerWHO,
        this.optimazerBO,
        this.outputs[1],
        this.hiddens[len - 1][1]
      );
      this.WHO = newW[0];
      this.BO = newW[1];
    } else {
      const newW = this.updateWightOutput(
        this.WHO,
        this.BO,
        this.optimazerWHO,
        this.optimazerBO,
        this.outputs[1],
        this.hiddenInput[0]
      );
      this.WHO = newW[0];
      this.BO = newW[1];
    }
    return { loss, err };
  }

  updateWHiddenLayer() {
    if (this.hiddenNodes.length !== 1) {
      for (let nodes = this.WHH.length - 1; nodes >= 0; nodes--) {
        if (nodes === this.WHH.length - 1 && nodes !== 0) {
          const newW = this.updateWight(
            this.WHO,
            this.WHH[nodes],
            this.BH[nodes],
            this.optimazerWHH[nodes],
            this.optimazerBH[nodes],
            this.hiddens[nodes][1],
            this.hiddens[nodes - 1][0]
          );
          this.WHH[nodes] = newW[0];
          this.BH[nodes] = newW[1];
        } else if (nodes !== 0) {
          const newW = this.updateWight(
            this.WHH[nodes + 1],
            this.WHH[nodes],
            this.BH[nodes],
            this.optimazerWHH[nodes],
            this.optimazerBH[nodes],
            this.hiddens[nodes][1],
            this.hiddens[nodes - 1][0]
          );
          this.WHH[nodes] = newW[0];
          this.BH[nodes] = newW[1];
        } else {
          if (this.WHH[nodes + 1] !== undefined) {
            const newW = this.updateWight(
              this.WHH[nodes + 1],
              this.WHH[nodes],
              this.BH[nodes],
              this.optimazerWHH[nodes],
              this.optimazerBH[nodes],
              this.hiddens[nodes][1],
              this.hiddenInput[0]
            );
            this.WHH[nodes] = newW[0];
            this.BH[nodes] = newW[1];
          } else {
            const newW = this.updateWight(
              this.WHO,
              this.WHH[nodes],
              this.BH[nodes],
              this.optimazerWHH[nodes],
              this.optimazerBH[nodes],
              this.hiddens[nodes][1],
              this.hiddenInput[0]
            );
            this.WHH[nodes] = newW[0];
            this.BH[nodes] = newW[1];
          }
        }
      }
    }
  }

  updateWInputLayer() {
    if (this.hiddenNodes.length !== 1) {
      const newW = this.updateWight(
        this.WHH[0],
        this.WIH,
        this.BI,
        this.optimazerWIH,
        this.optimazerBI,
        this.hiddenInput[1],
        this.inputs[0]
      );
      this.WIH = newW[0];
      this.BI = newW[1];
    } else {
      const newW = this.updateWight(
        this.WHO,
        this.WIH,
        this.BI,
        this.optimazerWIH,
        this.optimazerBI,
        this.hiddenInput[1],
        this.inputs[0]
      );
      this.WIH = newW[0];
      this.BI = newW[1];
    }
  }

  private updateWightOutput(
    WUp: mj.Matrix,
    BUp: mj.Matrix,
    optimazerW: optimazer,
    optimazerB: optimazer,
    dInput: mj.Matrix,
    input: mj.Matrix
  ) {
    const d_o_Err = mj.dotMultiply(dInput, this.updateErr);
    const d_o_lr = mj.multiply(d_o_Err, 1);
    const gWHO = mj.multiply(d_o_lr, mj.transpose(input));
    const optimazeW = optimazerW.optimazer(this.lr, gWHO);
    const optimazeB = optimazerB.optimazer(this.lr, d_o_lr);
    const newWIH = mj.subtract(WUp, optimazeW);
    const newB = mj.subtract(BUp, optimazeB);
    return [newWIH, newB];
  }

  private updateWight(
    wMul: mj.Matrix,
    WUp: mj.Matrix,
    BUp: mj.Matrix,
    optimazerW: optimazer,
    optimazerB: optimazer,
    dInput: mj.Matrix,
    input: mj.Matrix
  ) {
    const E = mj.multiply(mj.transpose(wMul), this.updateErr);
    const d_Err = mj.dotMultiply(dInput, E);
    const d_h_lr = mj.multiply(d_Err, 1);
    const gWIH = mj.multiply(d_h_lr, mj.transpose(input));
    const optimazeW = optimazerW.optimazer(this.lr, gWIH);
    const optimazeB = optimazerB.optimazer(this.lr, d_h_lr);
    const newWIH = mj.subtract(WUp, optimazeW);
    const newB = mj.subtract(BUp, optimazeB);
    this.updateErr = E;
    return [newWIH, newB];
  }

  private choiseOptimazer(shape: number[]) {
    let optimazer;
    switch (this.config.optimazer) {
      case "adaGrad":
        optimazer = new AdaGrad(shape);
        break;
      default:
        optimazer = new Sgd();
        break;
    }

    return optimazer;
  }

  predict(X: mj.Matrix): mj.Matrix {
    this.inputs = [X, X];

    this.calculateInputLayer();
    this.calculateHiddenLayer();
    const output = this.calculateOutputLayer();

    return output;
  }

  train(X: mj.Matrix, y: mj.Matrix): { loss: mj.MathNumericType | number } {
    this.inputs = [X, X];
    this.target = y;
    this.calculateInputLayer();
    this.calculateHiddenLayer();
    this.calculateOutputLayer();
    const { loss } = this.updateWOutputLayer();
    this.updateWHiddenLayer();
    this.updateWInputLayer();
    this.loss = loss;
    return { loss };
  }
}
