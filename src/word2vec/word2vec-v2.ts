import * as mj from "mathjs";
import Embedding from "./embedding";
import calculateActivation from "./calculateActivation";
import { activation } from "../activation/interfaces";
import calculateError from "../cost";
import { cost } from "../cost/interfaces";
import { cosineSimilarity } from "../utils/cosineSimilarity";

export default class W2V_V2 {
  embedding: Embedding;
  activation: activation;
  cost: cost;
  lr: number = 0.1;
  dataset: string[][];
  constructor(
    dataset: string[][],
    size: number,
    activation: activation,
    cost: cost
  ) {
    this.dataset = dataset;
    this.embedding = new Embedding(dataset, size);
    this.embedding.activation = activation;
    this.activation = activation;
    this.cost = cost;
  }

  wordSimilarity(word: string, topn: number = 10) {
    const input = this.embedding.getVector(word);
    const result = [];
    for (let word2 in this.embedding.corpus) {
      const target = this.embedding.getVector(word2);
      const similar = cosineSimilarity(target, input);
      if (word !== word2) result.push({ key: word2, similarity: similar });
    }
    const similarity = result.sort((a, b) => {
      return b.similarity - a.similarity;
    });

    for (let i = 0; i < topn; i++) {
      console.log(similarity[i]);
    }
  }

  findNeighbors(word: string, sentence: string[], windows: number) {
    const pos = sentence.indexOf(word);
    const neighbors: [string, mj.Matrix][] = [];
    for (let i = 1; i <= windows; i++) {
      const back = pos - i;
      const front = pos + i;
      if (back >= 0) {
        neighbors.push([
          sentence[back],
          this.embedding.getVector(sentence[back]),
        ]);
      }
      if (front < sentence.length) {
        neighbors.push([
          sentence[front],
          this.embedding.getVector(sentence[front]),
        ]);
      }
    }

    return neighbors;
  }

  skipGram(word: string, sentence: string[], windows: number) {
    let l: mj.MathNumericType = 1;
    const neighbors = this.findNeighbors(word, sentence, windows);
    if (neighbors[0] !== undefined) {
      for (let neighbor of neighbors) {
        const conteks = this.embedding.corpus[word];
        const input = mj.dotMultiply(conteks, neighbor[1]);

        const [result, dResult] = calculateActivation(input, this.activation);
        const [loss, err] = calculateError(neighbor[1], result, this.cost);
        l = loss;
        const dErrResult = mj.dotMultiply(dResult, err);
        const dErrResultLr = mj.dotMultiply(this.lr, dErrResult);

        const gConteks = mj.dotMultiply(dErrResultLr, neighbor[1]);
        const gNeighbor = mj.dotMultiply(dErrResultLr, conteks);

        const newConteks = mj.subtract(conteks, gConteks);
        const newNeighbor = mj.subtract(neighbor[1], gNeighbor);

        this.embedding.changeVector(word, newConteks);
        this.embedding.changeVector(neighbor[0], newNeighbor);
      }
    }
    return l;
  }

  cbow() {}

  setCorpus() {
    this.embedding.setCorpus();
  }

  fit(epochs: number, windows: number, sg = true) {
    let l;
    for (let i = 0; i < epochs; i++) {
      for (let sentence of this.dataset) {
        for (let word of sentence) {
          if (sg) {
            l = this.skipGram(word, sentence, windows);
          } else {
            // l = this.cbow(word, sentence, windows, activation);
          }
          console.clear();
          console.log(`epochs => ${i + 1}/${epochs}`);
          console.log("loss =>", l);
        }
      }
    }
  }
}
