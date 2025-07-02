import fs from "fs";
import { ELM } from "../src/core/ELM";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";

function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// 1️⃣ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// 2️⃣ Paragraphs
const paragraphs = rawText
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p.length > 30);
console.log(`✅ Parsed ${paragraphs.length} paragraphs.`);

// 3️⃣ Load supervised pairs
const supervisedCSV = fs.readFileSync("../public/supervised_pairs.csv", "utf8");
const pairs = supervisedCSV
    .split("\n")
    .slice(1)
    .map(l => l.split(",").map(s => s.trim()))
    .filter(r => r.length === 2);
console.log(`✅ Loaded ${pairs.length} supervised pairs.`);

// 4️⃣ Create encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyz0123456789",
    mode: "token",
    useTokenizer: true
});

// 5️⃣ Encode paragraphs
const paraVectors = paragraphs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p)))
);

// 6️⃣ Train/load unsupervised ELM
const unsupELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: 100,
    categories: [],
    log: { modelName: "UnsupervisedELM", verbose: true },
    dropout: 0.02
});
const unsupPath = "./elm_weights/unsupELM.json";
if (fs.existsSync(unsupPath)) {
    unsupELM.loadModelFromJSON(fs.readFileSync(unsupPath, "utf-8"));
    console.log("✅ Loaded UnsupervisedELM weights.");
} else {
    console.log("⚙️ Training UnsupervisedELM...");
    unsupELM.trainFromData(paraVectors, paraVectors);
    fs.writeFileSync(unsupPath, JSON.stringify(unsupELM.model));
    console.log("💾 Saved UnsupervisedELM weights.");
}

// 7️⃣ Encode supervised pairs
const queryVecs = pairs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p[0])))
);
const targetVecs = pairs.map(p =>
    l2normalize(encoder.normalize(encoder.encode(p[1])))
);

// 8️⃣ Train/load supervised ELM
const supELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: 100,
    categories: [],
    log: { modelName: "SupervisedELM", verbose: true },
    dropout: 0.02
});
const supPath = "./elm_weights/supELM.json";
if (fs.existsSync(supPath)) {
    supELM.loadModelFromJSON(fs.readFileSync(supPath, "utf-8"));
    console.log("✅ Loaded SupervisedELM weights.");
} else {
    console.log("⚙️ Training SupervisedELM...");
    supELM.trainFromData(queryVecs, targetVecs);
    fs.writeFileSync(supPath, JSON.stringify(supELM.model));
    console.log("💾 Saved SupervisedELM weights.");
}

// 9️⃣ Compute embeddings
const unsupEmb = unsupELM.computeHiddenLayer(paraVectors).map(l2normalize);
const paraEmb = paraVectors; // keep raw paragraph encoding for reference
const combinedEmb = unsupEmb.map((v, i) => l2normalize([...v, ...paraEmb[i]]));

// 🔟 Save embeddings
const records: EmbeddingRecord[] = paragraphs.map((p, i) => ({
    embedding: combinedEmb[i],
    metadata: { text: p }
}));
fs.mkdirSync("./embeddings", { recursive: true });
fs.writeFileSync("./embeddings/hybrid_embeddings.json", JSON.stringify(records, null, 2));
console.log("💾 Saved hybrid embeddings.");

// 🔍 Retrieval
function retrieve(query: string, topK = 5) {
    const qVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const supVec = supELM.computeHiddenLayer([qVec])[0];
    const unsupVec = unsupELM.computeHiddenLayer([qVec])[0];
    const combo = l2normalize([...unsupVec, ...supVec]);

    const scored = records.map(r => ({
        text: r.metadata.text,
        score: r.embedding.reduce((s, v, i) => s + v * combo[i], 0)
    }));
    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

const test = retrieve("How do you declare a map in Go?");
console.log("\n🔍 Retrieval results:");
test.forEach((r, i) =>
    console.log(` ${i + 1}. (${r.score.toFixed(4)}) ${r.text.slice(0, 80)}...`)
);

console.log("\n✅ Done.");
