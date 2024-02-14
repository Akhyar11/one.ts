import { Matrix, dot, dotDivide, dotMultiply, flatten, norm } from "mathjs";

export function cosineSimilarity(a: Matrix, b: Matrix) {
  const dotProduct = dot(flatten(a), flatten(b));
  const mA = norm(flatten(a));
  const mB = norm(flatten(b));
  const similarity = dotProduct / (Number(mA) * Number(mB));
  return similarity;
}
