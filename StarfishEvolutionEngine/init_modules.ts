import { ModulePool } from "../src/core/ModulePool";
import { ELM } from "../src/core/ELM";
import { MiniTransformer, Vocab } from "../src/core/MiniTransformer";
import fs from "fs";

const pool = new ModulePool();

// Load a corpus to initialize vocab
const rawText = fs.readFileSync("../public/go_textbook.md", "utf8");
const allChars = new Set<string>();
rawText.split("").forEach((c) => allChars.add(c.toLowerCase()));
const vocab = new Vocab(["<PAD>", "<UNK>", ...allChars]);

// Create 10 ELM modules
for (let i = 0; i < 5; i++) {
    pool.addModule({
        id: `elm-${i}`,
        type: "ELM",
        elm: new ELM({
            categories: [],
            hiddenUnits: 64 + Math.floor(Math.random() * 128),
            activation: ["relu", "tanh", "leakyRelu"][i % 3],
            maxLen: 2000,
            log: { modelName: `ELM-${i}`, verbose: false },
            dropout: Math.random() * 0.05,
        }),
        role: "retrieval",
        metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
        lastEvaluated: Date.now(),
    });
}

// Create 5 Transformer modules
for (let i = 0; i < 5; i++) {
    pool.addModule({
        id: `transformer-${i}`,
        type: "Transformer",
        transformer: new MiniTransformer(vocab),
        role: "retrieval",
        metrics: { recall1: 0, recall5: 0, mrr: 0, avgLatency: 0 },
        lastEvaluated: Date.now(),
    });
}

// Save initial module pool metadata
fs.writeFileSync("./module_pool.json", JSON.stringify(pool.listModules(), null, 2));
console.log("✅ Initialized module pool.");
