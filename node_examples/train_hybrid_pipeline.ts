import fs from "fs";
import { parse } from "csv-parse/sync";
import { ELM } from "../src/core/ELM";
import { ELMChain } from "../src/core/ELMChain";
import { UniversalEncoder } from "../src/preprocessing/UniversalEncoder";
import { TFIDFVectorizer } from "../src/core/TFIDF";
import { EmbeddingRecord } from "../src/core/EmbeddingStore";

// ✅ L2 normalization helper
function l2normalize(v: number[]): number[] {
    const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    return norm === 0 ? v : v.map(x => x / norm);
}

// ✅ Load corpus
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");

// ✅ Split paragraphs
const paragraphs = rawText
    .split(/\n{2,}/)
    .map(p => p.trim())
    .filter(p => p && p.length > 30);

console.log(`✅ Parsed ${paragraphs.length} paragraphs.`);

// ✅ Encoder
const encoder = new UniversalEncoder({
    maxLen: 100,
    charSet: "abcdefghijklmnopqrstuvwxyz0123456789",
    mode: "token",
    useTokenizer: true
});

// ✅ Encode paragraphs
const paraVectors = paragraphs.map(p =>
    encoder.normalize(encoder.encode(p))
);

// ✅ Load supervised pairs
const supervisedPaths = [
    "../public/supervised_pairs.csv",
    "../public/supervised_pairs_2.csv",
    "../public/supervised_pairs_3.csv",
    "../public/supervised_pairs_4.csv"
];
let supervisedPairs: { query: string, target: string }[] = [];

for (const path of supervisedPaths) {
    if (fs.existsSync(path)) {
        const csv = fs.readFileSync(path, "utf8");
        const rows = parse(csv, { skip_empty_lines: true });
        supervisedPairs.push(
            ...rows.map((row: [string, string]) => ({ query: row[0].trim(), target: row[1].trim() }))
        );
    }
}

console.log(`✅ Loaded ${supervisedPairs.length} supervised pairs.`);

const supQueryVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.query))
);
const supTargetVecs = supervisedPairs.map(p =>
    encoder.normalize(encoder.encode(p.target))
);

// ✅ Load negative pairs
const negativePaths = [
    "../public/negative_pairs.csv",
    "../public/negative_pairs_2.csv",
    "../public/negative_pairs_3.csv",
    "../public/negative_pairs_4.csv"
];
let negativePairs: { query: string, target: string }[] = [];

for (const path of negativePaths) {
    if (fs.existsSync(path)) {
        const csv = fs.readFileSync(path, "utf8");
        const rows = parse(csv, { skip_empty_lines: true });
        negativePairs.push(
            ...rows.map((row: [string, string]) => ({ query: row[0].trim(), target: row[1].trim() }))
        );
    }
}

console.log(`✅ Loaded ${negativePairs.length} negative pairs.`);

const negQueryVecs = negativePairs.map(p =>
    encoder.normalize(encoder.encode(p.query))
);
const negTargetVecs = negativePairs.map(p =>
    encoder.normalize(encoder.encode(p.target))
);

// ✅ Unsupervised ELM
const unsupELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: paraVectors[0].length,
    categories: [],
    log: { modelName: "UnsupervisedELM", verbose: true },
    dropout: 0.02
});
const unsupPath = "./elm_weights/unsupervised.json";
if (fs.existsSync(unsupPath)) {
    unsupELM.loadModelFromJSON(fs.readFileSync(unsupPath, "utf-8"));
    console.log(`✅ Loaded Unsupervised ELM weights.`);
} else {
    console.log(`⚙️ Training Unsupervised ELM...`);
    unsupELM.trainFromData(paraVectors, paraVectors);
    fs.writeFileSync(unsupPath, JSON.stringify(unsupELM.model));
    console.log(`💾 Saved Unsupervised ELM weights.`);
}

// ✅ Supervised ELM
const supELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: supQueryVecs[0].length,
    categories: [],
    log: { modelName: "SupervisedELM", verbose: true },
    dropout: 0.02
});
const supPath = "./elm_weights/supervised.json";
if (fs.existsSync(supPath)) {
    supELM.loadModelFromJSON(fs.readFileSync(supPath, "utf-8"));
    console.log(`✅ Loaded Supervised ELM weights.`);
} else {
    console.log(`⚙️ Training Supervised ELM...`);
    supELM.trainFromData(supQueryVecs, supTargetVecs);
    fs.writeFileSync(supPath, JSON.stringify(supELM.model));
    console.log(`💾 Saved Supervised ELM weights.`);
}

// ✅ Negative ELM
const negELM = new ELM({
    activation: "relu",
    hiddenUnits: 128,
    maxLen: negQueryVecs[0].length,
    categories: [],
    log: { modelName: "NegativeELM", verbose: true },
    dropout: 0.02
});
const negPath = "./elm_weights/negative.json";
if (fs.existsSync(negPath)) {
    negELM.loadModelFromJSON(fs.readFileSync(negPath, "utf-8"));
    console.log(`✅ Loaded Negative ELM weights.`);
} else {
    console.log(`⚙️ Training Negative ELM...`);
    negELM.trainFromData(negQueryVecs, negTargetVecs);
    fs.writeFileSync(negPath, JSON.stringify(negELM.model));
    console.log(`💾 Saved Negative ELM weights.`);
}

// ✅ Compute paragraph embeddings
const unsupEmb = unsupELM.computeHiddenLayer(paraVectors).map(l2normalize);

// ✅ TFIDF vectors
console.log(`⏳ Computing TFIDF vectors...`);
const vectorizer = new TFIDFVectorizer(paragraphs, 3000);
const tfidfVectors = vectorizer.vectorizeAll().map(l2normalize);
console.log(`✅ TFIDF vectors ready.`);

// ✅ Prepare final embeddings
const combinedEmbeddings = unsupEmb; // Start with unsupEmb

// ✅ ELMChain to refine
const chainHiddenUnits = [256, 128];
let embeddings = combinedEmbeddings;

const chainELMs = chainHiddenUnits.map((h, i) =>
    new ELM({
        activation: "relu",
        hiddenUnits: h,
        maxLen: embeddings[0].length,
        categories: [],
        log: { modelName: `IndexerELM_${i}`, verbose: true },
        dropout: 0.02
    })
);

chainELMs.forEach((elm, i) => {
    const path = `./elm_weights/indexer_layer${i}.json`;
    if (fs.existsSync(path)) {
        elm.loadModelFromJSON(fs.readFileSync(path, "utf-8"));
        console.log(`✅ Loaded Indexer layer ${i}.`);
    } else {
        console.log(`⚙️ Training Indexer layer ${i}...`);
        elm.trainFromData(embeddings, embeddings);
        fs.writeFileSync(path, JSON.stringify(elm.model));
        console.log(`💾 Saved Indexer layer ${i}.`);
    }
    embeddings = elm.computeHiddenLayer(embeddings).map(l2normalize);
});

const indexerChain = new ELMChain(chainELMs);
console.log(`✅ Indexer chain ready.`);

// ✅ Save embeddings
const embeddingRecords: EmbeddingRecord[] = paragraphs.map((p, i) => ({
    embedding: embeddings[i],
    metadata: { text: p }
}));
fs.writeFileSync("./embeddings/embeddings.json", JSON.stringify(embeddingRecords, null, 2));
console.log(`💾 Saved embeddings.`);

// ✅ Retrieval
function retrieve(query: string, topK = 5) {
    const qVec = encoder.normalize(encoder.encode(query));
    const unsupVec = unsupELM.computeHiddenLayer([qVec])[0];
    const supVec = supELM.computeHiddenLayer([qVec])[0];
    const negVec = negELM.computeHiddenLayer([qVec])[0];

    let combined = [
        ...unsupVec,
        ...supVec,
        ...negVec.map(x => -0.5 * x)
    ];
    combined = l2normalize(combined);

    const finalVec = indexerChain.getEmbedding([combined])[0];
    const tfidfQ = l2normalize(vectorizer.vectorize(query));

    const scored = embeddingRecords.map((r, i) => ({
        text: r.metadata.text,
        dense: r.embedding.reduce((s, v, j) => s + v * finalVec[j], 0),
        tfidf: tfidfVectors[i].reduce((s, v, j) => s + v * tfidfQ[j], 0)
    }));

    return scored
        .sort((a, b) => b.dense - a.dense)
        .slice(0, topK);
}

// ✅ Example retrieval
const results = retrieve("How do you declare a map in Go?");
console.log(`\n🔍 Retrieval results:`);
results.forEach((r, i) =>
    console.log(
        ` ${i + 1}. (Dense=${r.dense.toFixed(4)})\n${r.text.slice(0, 120)}...`
    )
);
console.log(`\n✅ Done.`);
