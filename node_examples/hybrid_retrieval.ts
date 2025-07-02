import fs from "fs";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { TFIDFVectorizer } from "../src/core/TFIDF";

// Utility: L2 normalize
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// 1️⃣ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// 2️⃣ Split into paragraphs
const paragraphs = rawText
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p && p.length > 30);

console.log(`✅ Parsed ${paragraphs.length} paragraphs.`);

// 3️⃣ Prepare encoders
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyz0123456789",
    mode: "token",
    useTokenizer: true
});

// 4️⃣ Compute TFIDF vectors
console.log(`⏳ Computing TFIDF vectors...`);
const vectorizer = new TFIDFVectorizer(paragraphs, 3000);
const tfidfVectors = vectorizer.vectorizeAll().map(l2normalize);
console.log(`✅ TFIDF vectors ready.`);

// 5️⃣ Compute paragraph embeddings
const paraVectors = paragraphs.map(p => l2normalize(encoder.normalize(encoder.encode(p))));

// 6️⃣ Train/load a paragraph ELM autoencoder
const paraELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: paraVectors[0].length,
    categories: [],
    log: { modelName: "ParagraphELM", verbose: true },
    dropout: 0.02
});
const weightsPath = "./elm_weights/paragraphELM.json";
if (fs.existsSync(weightsPath)) {
    const saved = fs.readFileSync(weightsPath, "utf-8");
    paraELM.loadModelFromJSON(saved);
    console.log(`✅ Loaded ParagraphELM weights.`);
} else {
    console.log(`⚙️ Training ParagraphELM...`);
    paraELM.trainFromData(paraVectors, paraVectors, { reuseWeights: false });
    fs.mkdirSync("./elm_weights", { recursive: true });
    fs.writeFileSync(weightsPath, JSON.stringify(paraELM.model));
    console.log(`💾 Saved ParagraphELM weights.`);
}

// 7️⃣ Get ELM embeddings
let embeddings = paraELM.computeHiddenLayer(paraVectors).map(l2normalize);

// 8️⃣ Train/load an Indexer ELM chain
const hiddenUnitSequence = [256, 128];
const indexerELMs = hiddenUnitSequence.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i}`, verbose: true },
        dropout: 0.02
    })
);

indexerELMs.forEach((elm, i) => {
    const p = `./elm_weights/indexerELM_${i}.json`;
    if (fs.existsSync(p)) {
        const saved = fs.readFileSync(p, "utf-8");
        elm.loadModelFromJSON(saved);
        console.log(`✅ Loaded IndexerELM_${i}.`);
    } else {
        console.log(`⚙️ Training IndexerELM_${i}...`);
        elm.trainFromData(embeddings, embeddings, { reuseWeights: false });
        fs.writeFileSync(p, JSON.stringify(elm.model));
        console.log(`💾 Saved IndexerELM_${i} weights.`);
    }
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});

const indexerChain = new ELMChain(indexerELMs);
console.log(`✅ Indexer chain ready.`);

// 9️⃣ Save embeddings
const embeddingRecords: EmbeddingRecord[] = paragraphs.map((p, i) => ({
    embedding: embeddings[i],
    metadata: { text: p }
}));
fs.writeFileSync("./embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings.`);

// 🔍 Hybrid retrieval function
function retrieve(query: string, topK = 5) {
    // Dense embedding
    const paraVec = l2normalize(encoder.normalize(encoder.encode(query)));
    const paraE = paraELM.computeHiddenLayer([paraVec])[0];
    const denseVec = indexerChain.getEmbedding([l2normalize(paraE)])[0];

    // TFIDF vector
    const tfidfVec = l2normalize(vectorizer.vectorize(query));

    // Scoring
    const scored = embeddingRecords.map((r, i) => {
        const denseScore = r.embedding.reduce((s, v, j) => s + v * denseVec[j], 0);
        const tfidfScore = tfidfVectors[i].reduce((s, v, j) => s + v * tfidfVec[j], 0);
        return {
            text: r.metadata.text,
            denseScore,
            tfidfScore,
            totalScore: 0.7 * denseScore + 0.3 * tfidfScore
        };
    });

    return scored.sort((a, b) => b.totalScore - a.totalScore).slice(0, topK);
}

// 🔍 Example retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Hybrid retrieval results:`);
results.forEach((r, i) =>
    console.log(
        `${i + 1}. [Dense=${r.denseScore.toFixed(4)} | TFIDF=${r.tfidfScore.toFixed(4)}] ${r.text.slice(0, 80)}...`
    )
);

console.log(`✅ Done.`);
