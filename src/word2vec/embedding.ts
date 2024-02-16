import { readFileSync, writeFileSync } from "fs";
import * as mj from "mathjs";
import { activation } from "../activation/interfaces";
import calculateActivation from "./calculateActivation";
import { cosineSimilarity } from "../utils/cosineSimilarity";

export default class Embedding {
  vectorSize: number;
  corpus: { [word: string]: mj.Matrix } = {};
  activation: activation = "sigmoid";
  private dataset: string[][];
  private corpusSave: { [word: string]: mj.MathArray } = {};
  isBinary: boolean;
  index: number = 1;
  constructor(dataset: string[][], size: number, isBinary = false) {
    this.dataset = dataset;
    this.vectorSize = size;
    this.isBinary = isBinary;
  }

  private generateBit(value: number): number[][] {
    const val = Math.max(0, Math.min(value, Math.pow(2, this.vectorSize) - 1));
    let binaryString = val.toString(2).padStart(this.vectorSize, "0");
    let encodeArray = binaryString.split("").map((bit) => [parseInt(bit)]);
    return encodeArray;
  }

  private generateVector(shape: number[]): mj.Matrix {
    return mj.matrix(mj.random(shape, -1, 1));
  }

  private createVector(index: number): mj.Matrix {
    const vector = !this.isBinary
      ? calculateActivation(
          this.generateVector([this.vectorSize, 1]),
          this.activation
        )[0]
      : mj.matrix(this.generateBit(index));
    return vector;
  }

  private addWordToCorpus(word: string): void {
    if (!this.corpus[word]) {
      this.corpus[word] = this.createVector(this.index);
      this.corpusSave[word] = this.corpus[word].toArray();
      this.index++;
    }
  }

  wordSimilarity(word: string, topn: number = 10) {
    const input = this.getVector(word);
    const result = [];
    for (let word2 in this.corpus) {
      const target = this.getVector(word2);
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
    for (let word2 in this.corpus) {
      const target = this.getVector(word2);
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

  len() {
    this.index = Object.keys(this.corpus).length + 1;
    return this.index;
  }

  setCorpus(): void {
    for (let kalimat of this.dataset) {
      for (let kata of kalimat) {
        this.len();
        this.addWordToCorpus(kata);
      }
    }
  }

  getVector(word: string): mj.Matrix {
    const vector = this.corpus[word];
    return vector;
  }

  changeVector(word: string, value: mj.Matrix) {
    this.corpusSave[word] = value.toArray();
  }

  saveCorpus(path: string) {
    const data = this.corpusSave;
    const dataJson = JSON.stringify(data);
    writeFileSync(path, dataJson);
  }

  loadCorpus(path: string) {
    const dataJson = readFileSync(path, "utf-8");
    const data = JSON.parse(dataJson);
    for (let word in data) {
      this.corpus[word] = mj.matrix(data[word]);
      this.corpusSave[word] = this.corpus[word].toArray();
    }
  }
}
