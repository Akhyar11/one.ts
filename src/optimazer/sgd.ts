import * as mj from "mathjs";

export default class Sgd {
  optimazer(lr: number, gradien: mj.Matrix): mj.Matrix {
    const newGradien = mj.dotMultiply(lr, gradien);
    return newGradien;
  }
}
