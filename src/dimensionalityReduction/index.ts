import { MathArray, Matrix } from "mathjs";
import Sequential from "../sequential";
import { config } from "../sequential/interface";

import * as mj from "mathjs";

export default class DimensionalityReduction extends Sequential {
  private nodes: number = 0;
  constructor(config: config) {
    super(config);
  }

  encode(X: Matrix, nodes: number): Matrix {
    this.inputs = [X, X];
    this.nodes = nodes;
    this.calculateInputLayer();
    this.calculateHiddenLayer();
    const encode = this.hiddenAsOutputLayer(this.nodes);
    return encode;
  }

  decode(X: Matrix): Matrix {
    const output = this.hiddenAsInputLayer(this.nodes, X);

    return output;
  }

  fit(X: mj.Matrix[], epocs: number): void {
    let l: mj.MathNumericType | number = 0;
    for (let i = 0; i < epocs; i++) {
      for (let x of X) {
        const { loss } = this.train(x, x);
        l = loss;
      }
      console.clear();
      console.log(`epocs => ${i + 1}/${epocs}`);
      console.log("loss =>", l);
    }
  }
}
