export default function tokenizeWord(text: string): string[] {
  const words = text.toLowerCase();
  const result = words.match(/\b\w+\b/g) || [];
  return result;
}
