import { Matrix } from "mathjs";
import * as mj from "mathjs";

export default function clip(x: Matrix, minValue: number, maxValue: number) {
  const clipFunction = (value: number) =>
    mj.max(minValue, mj.min(maxValue, value));
  return mj.map(x, clipFunction);
}
