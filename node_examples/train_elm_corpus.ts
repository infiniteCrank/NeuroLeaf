import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";

// Utility: L2 normalize a vector
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// 1️⃣ Load your Markdown corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// 2️⃣ Simple paragraph splitting (you can improve this)
const paragraphs = rawText
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p && p.length > 30);

console.log(`✅ Parsed ${paragraphs.length} paragraphs.`);

// 3️⃣ Encode all paragraphs with UniversalEncoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyz0123456789",
    mode: "token",
    useTokenizer: true
});

const baseVectors = paragraphs.map(p =>
    encoder.normalize(encoder.encode(p))
);

console.log(`✅ Encoded all paragraphs.`);

// 4️⃣ Train Encoder ELM as an autoencoder (target = input)
const encoderELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: 100,
    categories: [],
    log: { modelName: "EncoderELM", verbose: true },
    dropout: 0.02
});

// Load saved weights if available
const encoderWeightsPath = "./elm_weights/encoderELM.json";
if (fs.existsSync(encoderWeightsPath)) {
    fs.mkdirSync("./elm_weights", { recursive: true });
    const saved = fs.readFileSync(encoderWeightsPath, "utf-8");
    encoderELM.loadModelFromJSON(saved);
    console.log(`✅ Loaded Encoder ELM weights.`);
} else {
    console.log(`⚙️ Training Encoder ELM...`);
    encoderELM.trainFromData(baseVectors, baseVectors, { reuseWeights: false });
    if (encoderELM.model) {
        const json = JSON.stringify(encoderELM.model);
        fs.mkdirSync("./elm_weights", { recursive: true });
        fs.writeFileSync(encoderWeightsPath, json);
        console.log(`💾 Saved Encoder ELM weights.`);
    }
}

// 5️⃣ Compute embeddings from Encoder ELM
let embeddings = encoderELM.computeHiddenLayer(baseVectors).map(l2normalize);
console.log(`✅ Computed encoder embeddings.`);

// 6️⃣ Train an Indexer ELM Chain
const hiddenUnitSequence = [256, 128, 64];
const indexerELMs = hiddenUnitSequence.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i + 1}`, verbose: true },
        dropout: 0.02,
        metrics: {
            accuracy: 0.001
        }
    })
);

// Load or train each layer
indexerELMs.forEach((elm, layerIdx) => {
    const weightsPath = `./elm_weights/indexerELM_layer${layerIdx}.json`;
    if (fs.existsSync(weightsPath)) {
        fs.mkdirSync("./elm_weights", { recursive: true });
        const saved = fs.readFileSync(weightsPath, "utf-8");
        elm.loadModelFromJSON(saved);
        console.log(`✅ Loaded Indexer ELM #${layerIdx + 1} weights.`);
    } else {
        console.log(`⚙️ Training Indexer ELM #${layerIdx + 1}...`);
        elm.trainFromData(embeddings, embeddings, { reuseWeights: false });
        if (elm.model) {
            const json = JSON.stringify(elm.model);
            fs.writeFileSync(weightsPath, json);
            console.log(`💾 Saved Indexer ELM #${layerIdx + 1} weights.`);
        }
    }
    // Recompute embeddings for the next layer
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});

const indexerChain = new ELMChain(indexerELMs);
console.log(`✅ Indexer ELM chain ready.`);

// 7️⃣ Save all embeddings to disk
const embeddingRecords: EmbeddingRecord[] = paragraphs.map((p, i) => ({
    embedding: embeddings[i],
    metadata: { text: p }
}));
fs.writeFileSync("./embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings to embeddings.json.`);

// 8️⃣ Example retrieval
function retrieve(query: string, topK = 5) {
    const queryVec = encoderELM.computeHiddenLayer([
        encoder.normalize(encoder.encode(query))
    ])[0];
    const finalVec = indexerChain.getEmbedding([l2normalize(queryVec)])[0];

    const scored = embeddingRecords.map(r => ({
        text: r.metadata.text,
        score: r.embedding.reduce((sum, v, i) => sum + v * finalVec[i], 0)
    }));

    return scored.sort((a, b) => b.score - a.score).slice(0, topK);
}

// Example retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(` ${i + 1}. (${r.score.toFixed(4)}) ${r.text.slice(0, 80)}...`)
);

console.log(`\n✅ Done.`);
