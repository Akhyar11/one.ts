import * as mj from "mathjs";

export default class AdaGrad {
  sumGradien: mj.Matrix;
  lr: number = 0.1;
  epsilon: number = 0.0001;
  constructor(shape: number[]) {
    this.sumGradien = mj.matrix(mj.zeros(shape));
  }

  optimazer(lr: number, gradien: mj.Matrix): mj.Matrix {
    const pow = mj.map(this.sumGradien, (val) =>
      val !== 0 ? mj.pow(val, 2) : 0
    );
    const addEpsilon = mj.map(pow, (val) => val + this.epsilon);
    const sqrtGradien = mj.map(addEpsilon, (val) =>
      val !== 0 ? mj.sqrt(val) : 0
    );
    const adaGrad = mj.dotDivide(lr, sqrtGradien);
    const newGradien = mj.dotMultiply(gradien, adaGrad);
    this.sumGradien = mj.add(this.sumGradien, gradien);
    return newGradien;
  }
}
