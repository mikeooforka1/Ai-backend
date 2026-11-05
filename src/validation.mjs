export function selection(value) {
  if (
    !value ||
    !["x", "y", "w", "h"].every(
      (k) => typeof value[k] === "number" && Number.isFinite(value[k]),
    )
  )
    throw new Error("Selection must contain numeric x, y, w, h.");
  const { x, y, w, h } = value;
  if (
    x < 0 ||
    y < 0 ||
    w < 0.02 ||
    h < 0.02 ||
    x + w > 1.000001 ||
    y + h > 1.000001
  )
    throw new Error(
      "Select a box inside the frame, at least 2% wide and high.",
    );
  return { x, y, w, h };
}
export function mode(value) {
  if (!["object", "person", "face", "face_lipsync"].includes(value))
    throw new Error("Choose object, person, face or face_lipsync mode.");
  return value;
}

export function faceQuality(value = "original") {
  if (!["original", "natural"].includes(value)) throw new Error("Choose original or natural face blend.");
  return value;
}
