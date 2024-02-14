import { activation } from "../activation/interfaces";
import Sequential from "../sequential";
import { config } from "../sequential/interface";
import { cosineSimilarity } from "../utils/cosineSimilarity";
import Embedding from "./embedding";
import * as mj from "mathjs";

export default class W2V extends Sequential {
  embedding: Embedding;
  size: number = 64;
  dataset: string[][];
  nodes: number = 0;
  constructor(
    dataset: string[][],
    inputNodes: number = 20,
    size: number = 64,
    activation: [activation, activation] = ["tanh", "linear"],
    binary = false,
    config: config
  ) {
    super(config);
    this.dataset = dataset;
    this.embedding = new Embedding(dataset, inputNodes, binary);
    this.embedding.activation = activation[0];
    this.inputLayer(inputNodes);
    this.hiddenLayer(size, activation[1]);
    this.outputLayer(inputNodes, activation[0]);
  }

  findNeighbor(word: string, sentence: string[], windows: number): mj.Matrix[] {
    const pos = sentence.indexOf(word);
    const neighbors: mj.Matrix[] = [];
    for (let i = 1; i <= windows; i++) {
      const back = pos - i;
      const front = pos + i;
      if (back >= 0) {
        neighbors.push(this.embedding.getVector(sentence[back]));
      }
      if (front < sentence.length) {
        neighbors.push(this.embedding.getVector(sentence[front]));
      }
    }

    return neighbors;
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

  vectorSimilarity(vector: mj.Matrix, topn: number = 10) {
    const result = [];
    for (let word2 in this.embedding.corpus) {
      const target = this.embedding.getVector(word2);
      const similar = cosineSimilarity(target, vector);
      result.push({ key: word2, similarity: similar });
    }
    const similarity = result.sort((a, b) => {
      return b.similarity - a.similarity;
    });

    for (let i = 0; i < topn; i++) {
      console.log(similarity[i]);
    }
  }

  updateWordVector() {
    for (let sentence of this.dataset) {
      for (let word of sentence) {
        const oldVector = this.embedding.getVector(word);
        const newVector = this.predict(oldVector);
        this.embedding.changeVector(word, newVector);
      }
    }
  }

  encode(word: string): mj.Matrix {
    const X = this.embedding.getVector(word);
    this.inputs = [X, X];
    this.calculateInputLayer();
    this.calculateHiddenLayer();
    const encode = this.hiddenAsOutputLayer(this.nodes);
    return encode;
  }

  decode(X: mj.Matrix): mj.Matrix {
    const output = this.hiddenAsInputLayer(this.nodes, X);
    return output;
  }

  private cbow(word: string, sentence: string[], windows: number) {
    let l: mj.MathNumericType = 1;
    const target = this.embedding.corpus[word];
    const neighbors = this.findNeighbor(word, sentence, windows);
    if (neighbors[0] !== undefined) {
      let zero = mj.zeros(neighbors[0].size());
      for (let neighbor of neighbors) {
        zero = mj.add(zero, neighbor);
      }
      const input = mj.dotDivide(zero, neighbors.length);
      const { loss } = this.train(mj.matrix(input), target);
      l = loss;
    }
    return l;
  }

  private skipGram(word: string, sentence: string[], windows: number) {
    let l: mj.MathNumericType = 0;
    const input = this.embedding.corpus[word];
    const neighbors = this.findNeighbor(word, sentence, windows);
    if (neighbors[0] !== undefined) {
      for (let neighbor of neighbors) {
        const { loss } = this.train(input, neighbor);
        l = loss;
      }
    }
    return l;
  }

  saveEncode(path: string) {
    const keys = Object.keys(this.embedding.corpus);
    let index = 0;
    for (let word in this.embedding.corpus) {
      const encode = this.encode(word);
      this.embedding.changeVector(word, encode);
      index++;
      console.clear();
      console.log(`Jumlah kata dalam korpus => ${index}/${keys.length}`);
    }
    this.embedding.saveCorpus(path);
  }

  continueFit(
    modelPath: string,
    corpusPath: string,
    epochs: number,
    windows: number,
    sg = true
  ) {
    this.loadModels(modelPath);
    this.embedding.loadCorpus(corpusPath);
    this.fit(epochs, windows, sg, modelPath);
    this.saveModels(modelPath);
  }

  fit(epochs: number, windows: number, sg = true, saveModel: string = "") {
    let l;
    this.embedding.setCorpus();
    for (let i = 0; i < epochs; i++) {
      for (let sentence of this.dataset) {
        for (let word of sentence) {
          if (sg) {
            l = this.skipGram(word, sentence, windows);
          } else {
            l = this.cbow(word, sentence, windows);
          }
          console.clear();
          console.log(`epochs => ${i + 1}/${epochs}`);
          console.log("loss =>", l);
        }
        if (saveModel !== "") this.saveModels(saveModel);
      }
    }
  }
}
