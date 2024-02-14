import { dataStopWord } from "../../data/data";

export default function preprocessor(
  text: string,
  sw: string = dataStopWord
): string[] {
  const dataSW = sw.split("\n");
  const stopWord = new Set(dataSW);
  let preprocessedText = text.replace(/[^a-zA-Z\s]/g, "");
  preprocessedText = preprocessedText.toLowerCase();
  const tokens = preprocessedText.split(/\s+/);
  const removeStopWord = tokens.filter((token) => !stopWord.has(token));
  return removeStopWord;
}
